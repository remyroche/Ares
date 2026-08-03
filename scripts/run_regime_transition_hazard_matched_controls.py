#!/usr/bin/env python3
"""Matched-control challenger for the grouped transition hazard model.

For every training fold, pre-onset rows are contrasted with stable controls
matched on current state, nearby calendar period, state age, volatility,
breadth/fragmentation and trend direction.  Matching uses training labels only;
validation remains the full untouched grouped fold.  Selected controls receive
representation weights so down-sampling does not redefine the class prior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_transition_hazard import (
    HAZARD_HORIZONS,
    _model,
    build_transition_hazard_labels,
    causal_feature_columns,
    cumulative_metrics,
    event_recall_at_false_alerts,
    expand_at_risk_rows,
    predict_cumulative_hazard,
    stratified_event_recall,
)


DEFAULT_DATASET = (
    "data_perp/artifacts/regime_transition_research_20260726_v3/"
    "hourly_transition_dataset.parquet"
)
DEFAULT_EVENTS = (
    "data_perp/artifacts/regime_transition_research_20260726_v3/"
    "transition_events.parquet"
)
DEFAULT_BASELINE = (
    "data_perp/artifacts/regime_transition_hazard_challenger_20260727_v1/"
    "grouped_oof_cumulative_probabilities.parquet"
)
MATCH_FIELDS = (
    "state_context__state_age_hours",
    "peer_volatility_decoupling",
    "breadth_dispersion",
    "negative_breadth_pct",
    "btc_over_eth_dominance_roc",
)


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_safe(item) for item in value.tolist()]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def matched_training_base(
    frame: pd.DataFrame,
    event_time_hours: np.ndarray,
    train_indices: Sequence[int],
    *,
    controls_per_positive: int,
    calendar_radius_days: int,
    match_fields: Sequence[str] = MATCH_FIELDS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return all event rows, matched stable controls and representation weights."""
    train = np.asarray(train_indices, dtype=np.int64)
    event_time = np.asarray(event_time_hours, dtype=float)
    positive = train[np.isfinite(event_time[train])]
    control = train[~np.isfinite(event_time[train])]
    if not len(positive) or not len(control):
        raise ValueError("matched training requires event and stable rows")
    missing = sorted(
        {"source_utc", "state_context__current_state", *match_fields}
        - set(frame.columns)
    )
    if missing:
        raise ValueError(f"matching fields missing: {missing}")

    numeric = frame.loc[:, list(match_fields)].apply(
        pd.to_numeric, errors="coerce"
    )
    train_numeric = numeric.iloc[train]
    median = train_numeric.median()
    scale = (train_numeric.quantile(0.75) - train_numeric.quantile(0.25)).replace(
        0.0, np.nan
    )
    scale = scale.fillna(1.0)
    standardized = ((numeric - median) / scale).fillna(0.0).to_numpy(float)
    state = pd.to_numeric(
        frame["state_context__current_state"], errors="coerce"
    ).fillna(-1).astype(int).to_numpy()
    timestamp = pd.to_datetime(frame["source_utc"], utc=True, errors="raise")

    selected: list[int] = []
    distances: list[float] = []
    calendar_fallback_rows = 0
    state_fallback_rows = 0
    for state_value in np.unique(state[positive]):
        local_positive = positive[state[positive] == state_value]
        local_control = control[state[control] == state_value]
        if not len(local_control):
            local_control = control
            state_fallback_rows += int(len(local_positive))
        control_time = timestamp.iloc[local_control].to_numpy(
            dtype="datetime64[ns]"
        )
        for owner in local_positive:
            delta_days = np.abs(
                (control_time - timestamp.iloc[owner].to_datetime64())
                / np.timedelta64(1, "D")
            )
            pool_mask = delta_days <= float(calendar_radius_days)
            pool = local_control[pool_mask]
            if len(pool) < int(controls_per_positive):
                pool = local_control
                calendar_fallback_rows += 1
            take = min(int(controls_per_positive), len(pool))
            neighbour = NearestNeighbors(
                n_neighbors=take, metric="manhattan"
            ).fit(standardized[pool])
            distance, indices = neighbour.kneighbors(
                standardized[[owner]], return_distance=True
            )
            selected.extend(pool[indices[0]].tolist())
            distances.extend(distance[0].tolist())

    chosen_control = np.unique(np.asarray(selected, dtype=np.int64))
    chosen = np.sort(np.concatenate([positive, chosen_control]))
    representation = np.ones(len(frame), dtype=np.float32)
    for state_value in np.unique(state[control]):
        population = control[state[control] == state_value]
        sample = chosen_control[state[chosen_control] == state_value]
        if len(sample):
            representation[sample] = np.float32(len(population) / len(sample))
    return chosen, representation, {
        "train_event_rows": int(len(positive)),
        "train_stable_rows": int(len(control)),
        "matched_unique_control_rows": int(len(chosen_control)),
        "controls_per_positive": int(controls_per_positive),
        "calendar_radius_days": int(calendar_radius_days),
        "calendar_fallback_event_rows": int(calendar_fallback_rows),
        "state_fallback_event_rows": int(state_fallback_rows),
        "median_match_distance": float(np.median(distances)) if distances else np.nan,
        "p90_match_distance": float(np.quantile(distances, 0.90)) if distances else np.nan,
        "represented_stable_weight": float(representation[chosen_control].sum()),
    }


def baseline_prediction(
    frame: pd.DataFrame, baseline: pd.DataFrame
) -> np.ndarray:
    source = pd.to_datetime(frame["source_utc"], utc=True, errors="raise")
    local = baseline.copy()
    local["source_utc"] = pd.to_datetime(
        local["source_utc"], utc=True, errors="raise"
    )
    columns = [f"p_onset_within_{hours}h" for hours in HAZARD_HORIZONS]
    joined = pd.DataFrame({"source_utc": source}).merge(
        local[["source_utc", *columns]],
        on="source_utc",
        how="left",
        validate="one_to_one",
    )
    return joined[columns].to_numpy(float)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset)
    events_path = Path(args.events)
    baseline_path = Path(args.baseline)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    frame = pd.read_parquet(dataset_path)
    events = pd.read_parquet(events_path)
    features = causal_feature_columns(frame)
    matrix = frame.loc[:, features].apply(
        pd.to_numeric, errors="coerce"
    ).astype(np.float32)
    labels = build_transition_hazard_labels(
        frame, events=events, severity_weight=float(args.severity_weight)
    )
    base = np.flatnonzero(labels.base_mask)
    y_group = np.isfinite(labels.event_time_hours[base]).astype(np.int8)
    groups = labels.group_ids[base]
    splitter = StratifiedGroupKFold(
        n_splits=int(args.folds), shuffle=True, random_state=int(args.seed)
    )
    matched_oof = np.full(
        (len(frame), len(HAZARD_HORIZONS)), np.nan, dtype=np.float32
    )
    fold_rows: list[dict[str, Any]] = []
    for fold, (train_local, valid_local) in enumerate(
        splitter.split(base, y_group, groups=groups)
    ):
        train_full = base[train_local]
        valid = base[valid_local]
        train, representation, match_report = matched_training_base(
            frame,
            labels.event_time_hours,
            train_full,
            controls_per_positive=int(args.controls_per_positive),
            calendar_radius_days=int(args.calendar_radius_days),
        )
        train_x, train_y, owner, train_weight = expand_at_risk_rows(
            matrix, train, labels
        )
        train_weight = train_weight * representation[owner]
        model = _model(int(args.seed) + fold)
        model.fit(train_x, train_y, sample_weight=train_weight)
        matched_oof[valid] = predict_cumulative_hazard(model, matrix, valid)
        fold_rows.append(
            {
                "fold": fold,
                "full_train_base_rows": int(len(train_full)),
                "matched_train_base_rows": int(len(train)),
                "validation_base_rows": int(len(valid)),
                "expanded_train_rows": int(len(train_x)),
                "group_overlap": int(
                    len(set(labels.group_ids[train_full]).intersection(labels.group_ids[valid]))
                ),
                **match_report,
            }
        )
    baseline_oof = baseline_prediction(frame, pd.read_parquet(baseline_path))
    base_mask = labels.base_mask
    if not np.isfinite(matched_oof[base_mask]).all():
        raise AssertionError("matched-control OOF prediction is incomplete")
    if not np.isfinite(baseline_oof[base_mask]).all():
        raise AssertionError("baseline OOF prediction is incomplete")

    matched_alerts = event_recall_at_false_alerts(
        frame, labels, matched_oof, horizon_hours=3
    )
    baseline_alerts = event_recall_at_false_alerts(
        frame, labels, baseline_oof, horizon_hours=3
    )
    matched_threshold = next(
        row["threshold"]
        for row in matched_alerts
        if row["false_alert_budget_per_30d"] == 2.0
    )
    baseline_threshold = next(
        row["threshold"]
        for row in baseline_alerts
        if row["false_alert_budget_per_30d"] == 2.0
    )
    metrics = []
    for arm, prediction in (
        ("baseline_full_controls", baseline_oof),
        ("matched_controls", matched_oof),
    ):
        for horizon, row in cumulative_metrics(labels, prediction).items():
            metrics.append({"arm": arm, "horizon": horizon, **row})
    alert_rows = []
    for arm, rows in (
        ("baseline_full_controls", baseline_alerts),
        ("matched_controls", matched_alerts),
    ):
        for row in rows:
            alert_rows.append({"arm": arm, **row})
    kind_rows = []
    for arm, prediction, threshold in (
        ("baseline_full_controls", baseline_oof, baseline_threshold),
        ("matched_controls", matched_oof, matched_threshold),
    ):
        by_kind = stratified_event_recall(
            labels, prediction, threshold, horizon_hours=3
        )
        kind_rows.extend(
            {
                "arm": arm,
                "false_alert_budget_per_30d": 2.0,
                "event_kind": kind,
                "event_recall": recall,
            }
            for kind, recall in by_kind.items()
        )

    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(fold_rows).to_csv(
        output / "fold_matching_provenance.csv", index=False
    )
    pd.DataFrame(metrics).to_csv(output / "horizon_metrics.csv", index=False)
    pd.DataFrame(alert_rows).to_csv(
        output / "event_recall_at_false_alerts.csv", index=False
    )
    pd.DataFrame(kind_rows).to_csv(
        output / "event_recall_by_kind.csv", index=False
    )
    oof = frame.loc[
        base_mask,
        [
            "source_utc",
            "segment_id",
            "target__event_id",
            "target__time_to_onset_hours",
        ],
    ].copy()
    for index, horizon in enumerate(HAZARD_HORIZONS):
        oof[f"baseline_p_onset_within_{horizon}h"] = baseline_oof[
            base_mask, index
        ]
        oof[f"matched_p_onset_within_{horizon}h"] = matched_oof[
            base_mask, index
        ]
    oof.to_parquet(
        output / "paired_grouped_oof_predictions.parquet",
        index=False,
        compression="zstd",
    )
    baseline_3h = next(
        row for row in metrics
        if row["arm"] == "baseline_full_controls"
        and row["horizon"] == "within_3h"
    )
    matched_3h = next(
        row for row in metrics
        if row["arm"] == "matched_controls"
        and row["horizon"] == "within_3h"
    )
    baseline_2alerts = next(
        row for row in baseline_alerts
        if row["false_alert_budget_per_30d"] == 2.0
    )
    matched_2alerts = next(
        row for row in matched_alerts
        if row["false_alert_budget_per_30d"] == 2.0
    )
    manifest = {
        "schema": "regime_transition_hazard_matched_controls_v1",
        "status": "GROUPED_OOF_MATCHED_CONTROL_RESEARCH_COMPLETE",
        "promotion_eligible": False,
        "feature_count": len(features),
        "matching_contract": {
            "exact_fields": ["state_context__current_state"],
            "distance_fields": list(MATCH_FIELDS),
            "calendar_radius_days": int(args.calendar_radius_days),
            "controls_per_positive": int(args.controls_per_positive),
            "labels_used_only_for_training_match_selection": True,
            "validation_population": "full untouched grouped fold",
            "representation_weighting": (
                "selected stable controls reweighted within current state to "
                "preserve the full training stable-row mass"
            ),
        },
        "baseline_3h": baseline_3h,
        "matched_3h": matched_3h,
        "delta_3h_pr_auc": matched_3h["pr_auc"] - baseline_3h["pr_auc"],
        "baseline_2_alert_event_recall": baseline_2alerts["event_recall"],
        "matched_2_alert_event_recall": matched_2alerts["event_recall"],
        "delta_2_alert_event_recall": (
            matched_2alerts["event_recall"] - baseline_2alerts["event_recall"]
        ),
        "caveats": [
            "Grouped OOF uses pooled future state geometry and is not chronological promotion evidence.",
            "False-alert thresholds are selected on the same grouped OOF cohort.",
            "Matching changes training emphasis only; it does not authorize a hard onset gate.",
        ],
        "sources": {
            "dataset": {"path": str(dataset_path), "sha256": _sha256(dataset_path)},
            "events": {"path": str(events_path), "sha256": _sha256(events_path)},
            "baseline": {"path": str(baseline_path), "sha256": _sha256(baseline_path)},
        },
    }
    for filename in (
        "fold_matching_provenance.csv",
        "horizon_metrics.csv",
        "event_recall_at_false_alerts.csv",
        "event_recall_by_kind.csv",
        "paired_grouped_oof_predictions.parquet",
    ):
        manifest.setdefault("outputs", {})[filename] = {
            "path": str(output / filename),
            "sha256": _sha256(output / filename),
        }
    _write_json(output / "manifest.json", manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset", default=DEFAULT_DATASET)
    value.add_argument("--events", default=DEFAULT_EVENTS)
    value.add_argument("--baseline", default=DEFAULT_BASELINE)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--folds", type=int, default=5)
    value.add_argument("--seed", type=int, default=2219)
    value.add_argument("--severity-weight", type=float, default=0.25)
    value.add_argument("--controls-per-positive", type=int, default=3)
    value.add_argument("--calendar-radius-days", type=int, default=90)
    return value


if __name__ == "__main__":
    print(json.dumps(_safe(run(parser().parse_args())), indent=2, sort_keys=True))
