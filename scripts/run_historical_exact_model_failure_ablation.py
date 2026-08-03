#!/usr/bin/env python3
"""Grouped OOF ablation of market transition and exact model-health risk.

This operates only on the canonical February-April 2025 raw-alpha/exact-policy
lineage.  Failure-event windows remain whole within folds.  It compares market
state, compact model health, chronological active-transition probability and
explicit active-by-health interactions.  Results are research-only and do not
backfill the current execution-EV lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from materialize_historical_exact_model_health import HEALTH_COLUMNS


INTERACTION_HEALTH_COLUMNS = (
    "health__mapped_net_std",
    "health__raw_mapped_rank_abs_gap",
    "health__low_map_support_share",
    "health__selected_symbol_hhi",
    "health__recent_resolved_net_ev_hl3d",
    "health__recent_resolved_hit_rate_hl3d",
    "health__recent_resolved_mapping_error_hl3d",
    "health__recent_resolved_full_stop_rate_hl3d",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def market_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "source_utc",
        "execution_decision_utc",
        "segment_id",
        "target__pooled_state",
    }
    return [
        name
        for name in frame
        if name not in excluded
        and not name.startswith("target__")
        and pd.api.types.is_numeric_dtype(frame[name])
    ]


def add_active_health_interactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    work = frame.copy()
    probability = pd.to_numeric(
        work["active_transition_probability_oos"], errors="raise"
    )
    columns: list[str] = []
    for health_column in INTERACTION_HEALTH_COLUMNS:
        name = f"interaction__active_x__{health_column.removeprefix('health__')}"
        work[name] = probability * pd.to_numeric(
            work[health_column], errors="coerce"
        )
        columns.append(name)
    return work, columns


def failure_window_groups(
    frame: pd.DataFrame,
    *,
    target_column: str,
    event_column: str,
    window_hours: int = 12,
) -> np.ndarray:
    timestamp = pd.to_datetime(frame["source_utc"], utc=True, errors="raise")
    event_rows = frame.loc[
        frame[target_column].astype(bool) & frame[event_column].notna(),
        ["source_utc", event_column],
    ].copy()
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for _, local in event_rows.groupby(event_column, observed=True, sort=True):
        event_time = pd.to_datetime(local["source_utc"], utc=True)
        intervals.append(
            (
                event_time.min() - pd.Timedelta(hours=window_hours),
                event_time.max() + pd.Timedelta(hours=window_hours),
            )
        )
    intervals.sort()
    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    control = (
        "control_"
        + frame["segment_id"].astype(str)
        + "_"
        + timestamp.dt.tz_localize(None).dt.to_period("W").astype(str)
    ).to_numpy(object)
    groups = control.copy()
    for index, (start, end) in enumerate(merged):
        mask = timestamp.between(start, end, inclusive="both").to_numpy()
        groups[mask] = f"failure_cluster_{index:03d}"
    return groups.astype(str)


def _model(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=320,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.5,
        reg_lambda=8.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )


def grouped_oof(
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    target_column: str,
    event_column: str,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    x = frame[list(features)].apply(pd.to_numeric, errors="coerce")
    y = frame[target_column].astype(int).to_numpy()
    groups = failure_window_groups(
        frame, target_column=target_column, event_column=event_column
    )
    splitter = StratifiedGroupKFold(
        n_splits=int(folds), shuffle=True, random_state=int(seed)
    )
    prediction = np.full(len(frame), np.nan, dtype=np.float32)
    provenance: list[dict[str, Any]] = []
    for fold, (train, evaluation) in enumerate(
        splitter.split(x, y, groups=groups)
    ):
        overlap = set(groups[train]).intersection(groups[evaluation])
        if overlap:
            raise AssertionError("failure-window group leakage")
        model = _model(seed + fold)
        model.fit(x.iloc[train], y[train])
        prediction[evaluation] = model.predict_proba(x.iloc[evaluation])[:, 1]
        provenance.append(
            {
                "fold": fold,
                "train_rows": int(len(train)),
                "evaluation_rows": int(len(evaluation)),
                "train_positive_rows": int(y[train].sum()),
                "evaluation_positive_rows": int(y[evaluation].sum()),
                "group_overlap": 0,
                "train_end_utc": frame.iloc[train]["source_utc"].max(),
                "evaluation_start_utc": frame.iloc[evaluation]["source_utc"].min(),
                "evaluation_end_utc": frame.iloc[evaluation]["source_utc"].max(),
            }
        )
    if not np.isfinite(prediction).all():
        raise AssertionError("grouped OOF prediction is incomplete")
    return prediction, provenance, groups


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    return {
        "rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, prediction)),
        "brier": float(brier_score_loss(y, prediction)),
        "f1_at_0_5": float(
            f1_score(y, prediction >= 0.5, zero_division=0)
        ),
    }


def _episode_count(
    mask: np.ndarray, timestamps: pd.Series, *, refractory_hours: int = 6
) -> int:
    selected = pd.to_datetime(timestamps.loc[mask], utc=True).sort_values()
    if selected.empty:
        return 0
    return int(
        1
        + selected.diff()
        .gt(pd.Timedelta(hours=refractory_hours))
        .iloc[1:]
        .sum()
    )


def event_operating_curve(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    target_column: str,
    event_column: str,
    budgets: Sequence[float] = (1.0, 2.0, 4.0),
) -> list[dict[str, Any]]:
    y = frame[target_column].astype(bool).to_numpy()
    event_rows = frame.loc[y & frame[event_column].notna()].copy()
    event_rows["prediction"] = prediction[y & frame[event_column].notna()]
    event_max = event_rows.groupby(event_column, observed=True)[
        "prediction"
    ].max()
    negative = ~y
    days = max(
        float(
            (
                frame["source_utc"].max() - frame["source_utc"].min()
            )
            / pd.Timedelta(days=1)
        ),
        1.0,
    )
    thresholds = np.r_[
        np.inf, np.sort(np.unique(prediction[negative]))[::-1], -np.inf
    ]
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        limit = float(budget) * days / 30.0
        chosen = np.inf
        actual = 0
        for threshold in thresholds:
            count = _episode_count(
                prediction[negative] >= threshold,
                frame.loc[negative, "source_utc"],
            )
            if count <= limit:
                chosen = float(threshold)
                actual = int(count)
            else:
                break
        rows.append(
            {
                "false_alert_budget_per_30d": float(budget),
                "threshold": chosen if np.isfinite(chosen) else np.nan,
                "false_alerts_per_30d": float(actual * 30.0 / days),
                "event_count": int(len(event_max)),
                "event_recall": float(event_max.ge(chosen).mean()),
            }
        )
    return rows


def bootstrap_delta_ap(
    y: np.ndarray,
    baseline: np.ndarray,
    challenger: np.ndarray,
    groups: np.ndarray,
    *,
    draws: int = 1_000,
    seed: int = 20260729,
) -> dict[str, Any]:
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    delta: list[float] = []
    for _ in range(int(draws)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        local = np.concatenate([indices[group] for group in sampled])
        local_y = y[local]
        if len(np.unique(local_y)) < 2:
            continue
        delta.append(
            float(
                average_precision_score(local_y, challenger[local])
                - average_precision_score(local_y, baseline[local])
            )
        )
    values = np.asarray(delta, dtype=float)
    return {
        "bootstrap_draws": int(len(values)),
        "delta_pr_auc_p05": float(np.quantile(values, 0.05)),
        "delta_pr_auc_p50": float(np.quantile(values, 0.50)),
        "delta_pr_auc_p95": float(np.quantile(values, 0.95)),
        "probability_delta_pr_auc_positive": float(np.mean(values > 0.0)),
    }


def risk_tail_economics(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    target_column: str,
    fraction: float = 0.10,
) -> dict[str, Any]:
    count = max(1, int(np.ceil(float(fraction) * len(frame))))
    order = np.lexsort(
        (
            pd.to_datetime(frame["source_utc"], utc=True).astype("int64"),
            -prediction,
        )
    )
    selected = frame.iloc[order[:count]]
    return {
        "risk_tail_rows": int(len(selected)),
        "risk_tail_failure_rate": float(
            selected[target_column].astype(int).mean()
        ),
        "risk_tail_post_12h_net_bps": float(
            10_000.0 * selected["post_12h_net_ev_mean"].mean()
        ),
        "risk_tail_post_minus_pre_residual_bps": float(
            10_000.0
            * selected["post_minus_pre_mapping_residual"].mean()
        ),
        "risk_tail_mean_active_probability": float(
            selected["active_transition_probability_oos"].mean()
        ),
    }


def transition_failure_overlap(
    failures: pd.DataFrame,
    transitions: pd.DataFrame,
    *,
    horizon_hours: int = 6,
) -> pd.DataFrame:
    transition_time = pd.to_datetime(
        transitions["anchor_source_utc"], utc=True
    )
    rows: list[dict[str, Any]] = []
    for _, event in failures.iterrows():
        anchor = pd.Timestamp(event["anchor_source_utc"])
        distance = (
            (transition_time - anchor).abs() / pd.Timedelta(hours=1)
        )
        nearest = float(distance.min()) if len(distance) else np.nan
        rows.append(
            {
                "failure_label": event["failure_label"],
                "economic_event_id": event["economic_event_id"],
                "anchor_source_utc": anchor,
                "nearest_transition_distance_hours": nearest,
                "transition_within_6h": bool(nearest <= horizon_hours),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    health_path = Path(args.health)
    market_path = Path(args.market)
    active_path = Path(args.active_oos)
    failure_events_path = Path(args.failure_events)
    transitions_path = Path(args.transitions)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    health = pd.read_parquet(health_path)
    market = pd.read_parquet(market_path)
    active = pd.read_parquet(active_path)
    for local in (health, market, active):
        local["source_utc"] = pd.to_datetime(
            local["source_utc"], utc=True, errors="raise"
        )
    if health["source_utc"].duplicated().any():
        raise ValueError("health must have one row per source hour")
    active = active.rename(
        columns={"prediction": "active_transition_probability_oos"}
    )
    frame = health.merge(
        market, on="source_utc", how="inner", validate="one_to_one"
    ).merge(
        active[["source_utc", "active_transition_probability_oos"]],
        on="source_utc",
        how="inner",
        validate="one_to_one",
    )
    frame, interaction_columns = add_active_health_interactions(frame)
    market_features = market_feature_columns(market)
    health_features = list(HEALTH_COLUMNS)
    sets = {
        "market_only": market_features,
        "health_only": health_features,
        "active_only": ["active_transition_probability_oos"],
        "market_plus_health": market_features + health_features,
        "market_plus_health_plus_active": market_features
        + health_features
        + ["active_transition_probability_oos"],
        "market_plus_health_plus_active_interactions": market_features
        + health_features
        + ["active_transition_probability_oos"]
        + interaction_columns,
    }
    output.mkdir(parents=True, exist_ok=False)
    metric_rows: list[dict[str, Any]] = []
    operating_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    prediction_output = frame[
        [
            "source_utc",
            "post_12h_net_ev_mean",
            "post_minus_pre_mapping_residual",
            "active_transition_probability_oos",
            "target__economic_failure_broad_active",
            "target__economic_failure_broad_event_id",
            "target__economic_failure_strict_active",
            "target__economic_failure_strict_event_id",
        ]
    ].copy()
    fold_report: dict[str, Any] = {}
    for target_index, label in enumerate(("broad", "strict")):
        target = f"target__economic_failure_{label}_active"
        event = f"target__economic_failure_{label}_event_id"
        y = frame[target].astype(int).to_numpy()
        predictions: dict[str, np.ndarray] = {}
        groups_for_bootstrap: np.ndarray | None = None
        for set_index, (name, features) in enumerate(sets.items()):
            prediction, provenance, groups = grouped_oof(
                frame,
                features=features,
                target_column=target,
                event_column=event,
                folds=int(args.folds),
                seed=int(args.seed) + 100 * target_index + 11 * set_index,
            )
            predictions[name] = prediction
            groups_for_bootstrap = groups
            column = f"prediction__{label}__{name}"
            prediction_output[column] = prediction
            metric_rows.append(
                {
                    "failure_label": label,
                    "feature_set": name,
                    "feature_count": int(len(features)),
                    **_metrics(y, prediction),
                }
            )
            fold_report[f"{label}__{name}"] = provenance
            for row in event_operating_curve(
                frame,
                prediction,
                target_column=target,
                event_column=event,
            ):
                operating_rows.append(
                    {
                        "failure_label": label,
                        "feature_set": name,
                        **row,
                    }
                )
            for month, local in frame.assign(
                prediction=prediction
            ).groupby(
                frame["source_utc"].dt.strftime("%Y-%m"), sort=True
            ):
                local_y = local[target].astype(int).to_numpy()
                local_prediction = local["prediction"].to_numpy(float)
                monthly_rows.append(
                    {
                        "failure_label": label,
                        "feature_set": name,
                        "month": month,
                        **(
                            _metrics(local_y, local_prediction)
                            if len(np.unique(local_y)) == 2
                            else {
                                "rows": int(len(local_y)),
                                "positive_rows": int(local_y.sum()),
                                "prevalence": float(local_y.mean()),
                                "pr_auc": np.nan,
                                "roc_auc": np.nan,
                                "brier": float(
                                    brier_score_loss(
                                        local_y, local_prediction
                                    )
                                ),
                                "f1_at_0_5": float(
                                    f1_score(
                                        local_y,
                                        local_prediction >= 0.5,
                                        zero_division=0,
                                    )
                                ),
                            }
                        ),
                    }
                )
            tail_rows.append(
                {
                    "failure_label": label,
                    "feature_set": name,
                    **risk_tail_economics(
                        frame, prediction, target_column=target
                    ),
                }
            )
        baseline = predictions["market_only"]
        assert groups_for_bootstrap is not None
        for index, (name, prediction) in enumerate(predictions.items()):
            comparison = bootstrap_delta_ap(
                y,
                baseline,
                prediction,
                groups_for_bootstrap,
                seed=int(args.seed) + 1_000 * target_index + index,
            )
            for row in metric_rows:
                if (
                    row["failure_label"] == label
                    and row["feature_set"] == name
                ):
                    row.update(comparison)
                    break
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(operating_rows).to_csv(
        output / "event_operating_curve.csv", index=False
    )
    pd.DataFrame(monthly_rows).to_csv(
        output / "monthly_metrics.csv", index=False
    )
    pd.DataFrame(tail_rows).to_csv(
        output / "risk_tail_economics.csv", index=False
    )
    prediction_output.to_parquet(
        output / "grouped_oof_predictions.parquet",
        index=False,
        compression="zstd",
    )
    failures = pd.read_parquet(failure_events_path)
    transitions = pd.read_parquet(transitions_path)
    overlap = transition_failure_overlap(failures, transitions)
    overlap.to_csv(output / "failure_transition_overlap.csv", index=False)
    _write_json(output / "fold_provenance.json", fold_report)
    winner_rows = []
    for label, local in metrics.groupby("failure_label", sort=True):
        winner = local.sort_values(
            ["pr_auc", "brier"], ascending=[False, True]
        ).iloc[0]
        winner_rows.append(
            {
                "failure_label": label,
                "feature_set": winner["feature_set"],
                "pr_auc": float(winner["pr_auc"]),
                "roc_auc": float(winner["roc_auc"]),
                "brier": float(winner["brier"]),
                "latest_month_pr_auc": float(
                    pd.DataFrame(monthly_rows)
                    .loc[
                        lambda x: x["failure_label"].eq(label)
                        & x["feature_set"].eq(winner["feature_set"])
                        & x["month"].eq("2025-04"),
                        "pr_auc",
                    ]
                    .iloc[0]
                ),
            }
        )
    manifest = {
        "schema": "historical_exact_model_failure_ablation_v1",
        "status": "GROUPED_OOF_RESEARCH_COMPLETE",
        "promotion_eligible": False,
        "promotion_blocker": (
            "February-April raw-alpha lineage rather than current execution-EV; "
            "grouped OOF and pooled upstream state geometry; policy use untested"
        ),
        "lineage": "canonical February-April 2025 raw-alpha exact-policy",
        "validation_contract": (
            "failure active runs plus +/-12h context are merged into indivisible "
            "groups; remaining controls use segment-calendar-week blocks"
        ),
        "feature_sets": {
            name: len(features) for name, features in sets.items()
        },
        "winners": winner_rows,
        "failure_transition_overlap": {
            label: {
                "events": int(len(local)),
                "within_6h": int(local["transition_within_6h"].sum()),
            }
            for label, local in overlap.groupby("failure_label", sort=True)
        },
        "sources": {
            "health": {"path": str(health_path), "sha256": _sha256(health_path)},
            "market": {"path": str(market_path), "sha256": _sha256(market_path)},
            "active_oos": {
                "path": str(active_path),
                "sha256": _sha256(active_path),
            },
            "failure_events": {
                "path": str(failure_events_path),
                "sha256": _sha256(failure_events_path),
            },
            "transitions": {
                "path": str(transitions_path),
                "sha256": _sha256(transitions_path),
            },
        },
        "outputs": {},
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    output_paths = {
        name: output / filename
        for name, filename in {
            "metrics": "metrics.csv",
            "event_operating_curve": "event_operating_curve.csv",
            "monthly_metrics": "monthly_metrics.csv",
            "risk_tail_economics": "risk_tail_economics.csv",
            "predictions": "grouped_oof_predictions.parquet",
            "failure_transition_overlap": "failure_transition_overlap.csv",
            "fold_provenance": "fold_provenance.json",
        }.items()
    }
    manifest["outputs"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in output_paths.items()
    }
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    root = Path("/Users/remyroche/Documents/Ares")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--health",
        type=Path,
        default=root
        / (
            "data_perp/artifacts/historical_exact_model_health_failure_20260729_v3/"
            "hourly_exact_model_health_and_failure_labels.parquet"
        ),
    )
    parser.add_argument(
        "--market",
        type=Path,
        default=root
        / (
            "data_perp/artifacts/regime_transition_research_20260726_v3/"
            "hourly_transition_dataset.parquet"
        ),
    )
    parser.add_argument(
        "--active-oos",
        type=Path,
        default=root
        / (
            "data_perp/artifacts/regime_transition_active_head_chronological_oos_20260729_v2/"
            "chronological_oos.parquet"
        ),
    )
    parser.add_argument(
        "--failure-events",
        type=Path,
        default=root
        / (
            "data_perp/artifacts/historical_exact_model_health_failure_20260729_v3/"
            "economic_failure_events.parquet"
        ),
    )
    parser.add_argument(
        "--transitions",
        type=Path,
        default=root
        / (
            "data_perp/artifacts/regime_transition_research_20260726_v3/"
            "transition_events.parquet"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3929)
    return parser


def main() -> None:
    print(json.dumps(_safe(run(_parser().parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
