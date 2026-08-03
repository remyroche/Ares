"""Grouped discrete-time cumulative-hazard challenger for transition onset.

This is deliberately a research-only model for the canonical symmetric
transition panel.  A base decision row is expanded only over intervals for
which it remains observable and event-free.  The model learns conditional
hazards, then derives every cumulative onset probability as
``1 - prod(1 - hazard)``.  Therefore cumulative probabilities are monotone by
construction rather than repaired after scoring.

The label contract is:

* intervals ``(0, 1]``, ``(1, 3]``, ``(3, 6]``, ``(6, 12]`` hours;
* rows at or after an onset are not at risk;
* each canonical segment is checked for exact one-hour continuity, and a gap
  censors every preceding row at its last observed hour;
* each transition's pre-onset rows remain in one validation group; stable
  controls remain in seven-day blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


HAZARD_INTERVALS: tuple[tuple[int, int], ...] = ((0, 1), (1, 3), (3, 6), (6, 12))
HAZARD_HORIZONS: tuple[int, ...] = tuple(upper for _, upper in HAZARD_INTERVALS)
TRANSITION_HAZARD_SCHEMA = "grouped_transition_discrete_hazard_v1"


@dataclass(frozen=True)
class TransitionHazardLabels:
    """Base-row labels and observability data used to expand the risk set."""

    base_mask: np.ndarray
    event_time_hours: np.ndarray
    followup_hours: np.ndarray
    group_ids: np.ndarray
    event_ids: np.ndarray
    event_kind: np.ndarray
    severity: np.ndarray
    base_weight: np.ndarray


def causal_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return only numeric, decision-time feature columns from the v3 panel."""

    blocked = {"source_utc", "execution_decision_utc", "segment_id"}
    return [
        name
        for name in frame.columns
        if name not in blocked
        and not name.startswith(("target__", "expost__"))
        and pd.api.types.is_numeric_dtype(frame[name])
    ]


def _event_metadata(frame: pd.DataFrame, events: pd.DataFrame | None) -> pd.DataFrame:
    """Build severity and abrupt/gradual labels without using them as features."""

    event_id = frame["target__event_id"].dropna().astype(str)
    ids = pd.Index(event_id.unique(), name="event_id")
    result = pd.DataFrame(index=ids)
    result["severity"] = 1.0
    result["event_kind"] = "unknown"
    if events is None or not len(events):
        return result.reset_index()
    required = {"event_id", "anchor_source_utc", "transition_end_utc"}
    missing = required.difference(events.columns)
    if missing:
        raise KeyError(f"transition events missing {sorted(missing)}")
    event_frame = events.drop_duplicates("event_id").set_index("event_id")
    matched = event_frame.reindex(ids)
    duration = (
        pd.to_datetime(matched["transition_end_utc"], utc=True)
        - pd.to_datetime(matched["anchor_source_utc"], utc=True)
    ) / pd.Timedelta(hours=1)
    # The canonical active label ends after three persistent destination hours;
    # a short interval is therefore an abrupt transition, a longer one gradual.
    result["event_kind"] = np.where(duration.le(3.0), "abrupt", "gradual")
    if "robust_pre_post_shift" in matched:
        severity = pd.to_numeric(matched["robust_pre_post_shift"], errors="coerce")
        result["severity"] = severity.where(severity.gt(0.0), 1.0).fillna(1.0)
    return result.reset_index()


def _segment_followup_hours(frame: pd.DataFrame) -> np.ndarray:
    """Return exact observed follow-up before each segment end/gap.

    ``segment_id`` is expected to have been produced by the canonical loader,
    but we still split a segment whenever timestamp continuity is broken.  This
    avoids assigning a survival control across a missing bar.
    """

    stamp = pd.to_datetime(frame["source_utc"], utc=True)
    followup = np.zeros(len(frame), dtype=np.float32)
    for _, positions in frame.groupby("segment_id", observed=True, sort=False).groups.items():
        loc = np.asarray(list(positions), dtype=np.int64)
        loc = loc[np.argsort(stamp.iloc[loc].to_numpy())]
        local_stamp = stamp.iloc[loc].to_numpy(dtype="datetime64[ns]")
        run_start = 0
        for offset in range(1, len(loc) + 1):
            boundary = offset == len(loc)
            if not boundary:
                boundary = local_stamp[offset] - local_stamp[offset - 1] != np.timedelta64(1, "h")
            if boundary:
                run = loc[run_start:offset]
                followup[run] = np.arange(len(run) - 1, -1, -1, dtype=np.float32)
                run_start = offset
    return followup


def build_transition_hazard_labels(
    frame: pd.DataFrame,
    *,
    events: pd.DataFrame | None = None,
    severity_weight: float = 0.0,
    control_block_days: int = 7,
) -> TransitionHazardLabels:
    """Construct censor-aware labels from canonical v3 transition targets.

    Severity weighting is optional.  When enabled, all at-risk rows belonging
    to an event receive the same multiplier, preserving a coherent weighted
    risk set instead of weighting only the positive hazard row.
    """

    required = {
        "source_utc",
        "segment_id",
        "target__event_id",
        "target__time_to_onset_hours",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"transition frame missing {sorted(missing)}")
    if not 0.0 <= float(severity_weight) <= 1.0:
        raise ValueError("severity_weight must be between 0 and 1")
    if int(control_block_days) < 1:
        raise ValueError("control_block_days must be positive")

    n = len(frame)
    event_id = frame["target__event_id"].astype("string")
    relative = pd.to_numeric(frame["target__time_to_onset_hours"], errors="coerce")
    pre_event = event_id.notna() & relative.lt(0.0)
    stable = event_id.isna()
    # Only stable rows and the canonical [-12h, 0h) lead window are at risk.
    base_mask = (stable | pre_event).to_numpy(bool)
    event_time = np.full(n, np.inf, dtype=np.float32)
    event_time[pre_event.to_numpy()] = (-relative.loc[pre_event]).to_numpy(np.float32)
    followup = _segment_followup_hours(frame)

    metadata = _event_metadata(frame, events).set_index("event_id")
    event_key = event_id.fillna("").astype(str)
    severity = event_key.map(metadata["severity"]).fillna(1.0).to_numpy(np.float32)
    kind = event_key.map(metadata["event_kind"]).fillna("control").to_numpy(str)
    median = float(np.nanmedian(metadata["severity"].to_numpy(float))) if len(metadata) else 1.0
    scaled = np.clip(severity / max(median, 1e-6), 0.0, 4.0)
    weight = np.ones(n, dtype=np.float32)
    weight[pre_event.to_numpy()] = 1.0 + float(severity_weight) * (scaled[pre_event.to_numpy()] - 1.0)
    weight = np.clip(weight, 0.25, 5.0)

    stamp = pd.to_datetime(frame["source_utc"], utc=True)
    control_block = (
        "control_"
        + frame["segment_id"].astype(str)
        + "_"
        + (stamp.dt.floor(f"{int(control_block_days)}D").astype(str))
    )
    event_groups = np.char.add("event_", event_key.to_numpy(dtype=str))
    groups = np.where(pre_event.to_numpy(), event_groups, control_block.to_numpy(dtype=str))
    return TransitionHazardLabels(
        base_mask=base_mask,
        event_time_hours=event_time,
        followup_hours=followup,
        group_ids=groups.astype(str),
        event_ids=event_key.to_numpy(str),
        event_kind=kind,
        severity=severity,
        base_weight=weight,
    )


def expand_at_risk_rows(
    matrix: pd.DataFrame,
    base_indices: Sequence[int],
    labels: TransitionHazardLabels,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Expand base decisions into valid at-risk interval records.

    Returns expanded features, binary interval event labels, base-row indices,
    and per-record weights.  A censored base row contributes only intervals
    fully observed before the segment boundary.
    """

    base = np.asarray(base_indices, dtype=np.int64)
    pieces: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for interval, (lower, upper) in enumerate(HAZARD_INTERVALS):
        time = labels.event_time_hours[base]
        observed = labels.followup_hours[base]
        event_here = (time > lower) & (time <= upper)
        control_here = (time > upper) & (observed >= upper)
        keep = event_here | control_here
        owner = base[keep]
        if not len(owner):
            continue
        local = matrix.iloc[owner].reset_index(drop=True).copy()
        for number, (_, endpoint) in enumerate(HAZARD_INTERVALS):
            local[f"__hazard_interval_to_{endpoint}h"] = np.float32(number == interval)
        local["__hazard_interval_width_h"] = np.float32(upper - lower)
        pieces.append(local)
        targets.append(event_here[keep].astype(np.float32))
        owners.append(owner)
        weights.append(labels.base_weight[owner])
    if not pieces:
        raise ValueError("no fully observed at-risk intervals")
    return (
        pd.concat(pieces, ignore_index=True),
        np.concatenate(targets),
        np.concatenate(owners),
        np.concatenate(weights),
    )


def predict_cumulative_hazard(
    model: Any,
    matrix: pd.DataFrame,
    base_indices: Sequence[int],
) -> np.ndarray:
    """Score conditional hazards then return monotone cumulative probabilities."""

    base = np.asarray(base_indices, dtype=np.int64)
    hazards: list[np.ndarray] = []
    for interval, (_, upper) in enumerate(HAZARD_INTERVALS):
        local = matrix.iloc[base].reset_index(drop=True).copy()
        for number, (_, endpoint) in enumerate(HAZARD_INTERVALS):
            local[f"__hazard_interval_to_{endpoint}h"] = np.float32(number == interval)
        local["__hazard_interval_width_h"] = np.float32(HAZARD_INTERVALS[interval][1] - HAZARD_INTERVALS[interval][0])
        hazards.append(np.clip(model.predict_proba(local)[:, 1], 0.0, 1.0))
    return 1.0 - np.cumprod(1.0 - np.column_stack(hazards), axis=1)


def _model(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=320,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=35,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.25,
        reg_lambda=8.0,
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )


def _safe_binary_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    if len(y) == 0 or np.unique(y).size < 2:
        return {"pr_auc": None, "roc_auc": None, "brier": None, "rows": int(len(y)), "prevalence": float(np.mean(y)) if len(y) else None}
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
        "rows": int(len(y)),
        "prevalence": float(np.mean(y)),
    }


def cumulative_metrics(labels: TransitionHazardLabels, prediction: np.ndarray) -> dict[str, dict[str, float | None]]:
    """Evaluate each horizon only where an outcome is observed or occurs."""

    mask = labels.base_mask
    event = labels.event_time_hours
    result: dict[str, dict[str, float | None]] = {}
    for column, hours in enumerate(HAZARD_HORIZONS):
        valid = mask & ((event <= hours) | (labels.followup_hours >= hours))
        y = (event[valid] <= hours).astype(np.int8)
        result[f"within_{hours}h"] = _safe_binary_metrics(y, prediction[valid, column])
    return result


def _alert_count(timestamp: pd.Series, segment: pd.Series, score: np.ndarray, threshold: float, refractory_hours: int) -> int:
    work = pd.DataFrame({"timestamp": pd.to_datetime(timestamp, utc=True), "segment": segment.to_numpy(), "score": score})
    alerts = 0
    for _, local in work.loc[work["score"] >= threshold].sort_values("timestamp").groupby("segment", sort=False):
        last: pd.Timestamp | None = None
        for stamp in local["timestamp"]:
            if last is None or stamp - last >= pd.Timedelta(hours=refractory_hours):
                alerts += 1
                last = stamp
    return alerts


def event_recall_at_false_alerts(
    frame: pd.DataFrame,
    labels: TransitionHazardLabels,
    prediction: np.ndarray,
    *,
    horizon_hours: int = 3,
    false_alerts_per_30d: Sequence[float] = (1.0, 2.0, 4.0),
    refractory_hours: int = 6,
) -> list[dict[str, float | int | None]]:
    """Choose OOF thresholds under fixed false-alert budgets and report recall."""

    try:
        column = HAZARD_HORIZONS.index(int(horizon_hours))
    except ValueError as exc:
        raise ValueError(f"unknown horizon {horizon_hours}") from exc
    base = labels.base_mask
    score = prediction[:, column]
    event_time = labels.event_time_hours
    negative = base & (event_time > horizon_hours) & (labels.followup_hours >= horizon_hours)
    event = base & np.isfinite(event_time) & (event_time <= horizon_hours)
    work_start = pd.to_datetime(frame.loc[base, "source_utc"], utc=True).min()
    work_end = pd.to_datetime(frame.loc[base, "source_utc"], utc=True).max()
    days = max((work_end - work_start) / pd.Timedelta(days=1), 1.0)
    values = np.unique(score[negative])
    values = np.r_[np.inf, np.sort(values)[::-1], -np.inf]
    event_rows = pd.DataFrame({"event_id": labels.event_ids[event], "score": score[event]})
    event_max = event_rows.loc[event_rows["event_id"] != ""].groupby("event_id", observed=True)["score"].max()
    result: list[dict[str, float | int | None]] = []
    for budget in false_alerts_per_30d:
        cap = float(budget) * days / 30.0
        chosen = float("inf")
        actual = 0
        for threshold in values:
            count = _alert_count(frame.loc[negative, "source_utc"], frame.loc[negative, "segment_id"], score[negative], float(threshold), refractory_hours)
            if count <= cap:
                chosen, actual = float(threshold), count
            else:
                break
        detected = event_max.ge(chosen) if len(event_max) else pd.Series(dtype=bool)
        result.append({
            "horizon_hours": int(horizon_hours),
            "false_alert_budget_per_30d": float(budget),
            "threshold": chosen if np.isfinite(chosen) else None,
            "false_alerts_per_30d": float(actual / days * 30.0),
            "event_count": int(len(event_max)),
            "event_recall": float(detected.mean()) if len(detected) else None,
        })
    return result


def stratified_event_recall(
    labels: TransitionHazardLabels,
    prediction: np.ndarray,
    threshold: float | None,
    *,
    horizon_hours: int = 3,
) -> dict[str, float | None]:
    """Report abrupt/gradual event recall at one selected false-alert cutoff."""

    if threshold is None:
        return {"abrupt": None, "gradual": None}
    column = HAZARD_HORIZONS.index(int(horizon_hours))
    eligible = labels.base_mask & np.isfinite(labels.event_time_hours) & (labels.event_time_hours <= horizon_hours)
    work = pd.DataFrame({"event_id": labels.event_ids[eligible], "kind": labels.event_kind[eligible], "score": prediction[eligible, column]})
    work = work.loc[work["event_id"] != ""]
    if not len(work):
        return {"abrupt": None, "gradual": None}
    detected = work.groupby(["event_id", "kind"], observed=True)["score"].max().ge(threshold).reset_index()
    return {kind: float(detected.loc[detected["kind"].eq(kind), "score"].mean()) if detected["kind"].eq(kind).any() else None for kind in ("abrupt", "gradual")}


def fit_grouped_transition_hazard(
    frame: pd.DataFrame,
    *,
    events: pd.DataFrame | None = None,
    folds: int = 5,
    seed: int = 2219,
    severity_weight: float = 0.0,
) -> dict[str, Any]:
    """Fit grouped OOF and final discrete-hazard transition models."""

    if int(folds) < 2:
        raise ValueError("at least two folds are required")
    features = causal_feature_columns(frame)
    if not features:
        raise ValueError("no causal numeric features")
    matrix = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    labels = build_transition_hazard_labels(frame, events=events, severity_weight=severity_weight)
    base = np.flatnonzero(labels.base_mask)
    y_group = np.isfinite(labels.event_time_hours[base]).astype(np.int8)
    groups = labels.group_ids[base]
    distinct_positive = np.unique(groups[y_group > 0]).size
    if distinct_positive < int(folds):
        raise ValueError("not enough event groups for requested folds")
    splitter = StratifiedGroupKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    oof = np.full((len(frame), len(HAZARD_HORIZONS)), np.nan, dtype=np.float32)
    fold_ids = np.full(len(frame), -1, dtype=np.int16)
    fold_provenance: list[dict[str, Any]] = []
    oof_models: list[Any] = []
    for fold, (train_local, valid_local) in enumerate(splitter.split(base, y_group, groups=groups)):
        train, valid = base[train_local], base[valid_local]
        train_x, train_y, _, train_weight = expand_at_risk_rows(matrix, train, labels)
        model = _model(int(seed) + fold)
        model.fit(train_x, train_y, sample_weight=train_weight)
        oof[valid] = predict_cumulative_hazard(model, matrix, valid)
        fold_ids[valid] = fold
        oof_models.append(model)
        fold_provenance.append({
            "fold": fold,
            "train_base_rows": int(len(train)),
            "validation_base_rows": int(len(valid)),
            "train_event_groups": int(np.unique(labels.group_ids[train][np.isfinite(labels.event_time_hours[train])]).size),
            "validation_event_groups": int(np.unique(labels.group_ids[valid][np.isfinite(labels.event_time_hours[valid])]).size),
            "expanded_train_rows": int(len(train_x)),
            "group_overlap": int(len(set(labels.group_ids[train]).intersection(labels.group_ids[valid]))),
        })
    final_x, final_y, _, final_weight = expand_at_risk_rows(matrix, base, labels)
    final_model = _model(int(seed) + 10_000)
    final_model.fit(final_x, final_y, sample_weight=final_weight)
    alerts = event_recall_at_false_alerts(frame, labels, oof, horizon_hours=3)
    threshold = next((row["threshold"] for row in alerts if row["false_alert_budget_per_30d"] == 2.0), None)
    return {
        "schema": TRANSITION_HAZARD_SCHEMA,
        "features": features,
        "labels": labels,
        "oof_prediction": oof,
        "oof_fold_ids": fold_ids,
        "oof_models": oof_models,
        "final_model": final_model,
        "fold_provenance": fold_provenance,
        "metrics_by_horizon": cumulative_metrics(labels, oof),
        "event_recall_at_false_alerts": alerts,
        "event_recall_by_kind_at_2_false_alerts_per_30d": stratified_event_recall(labels, oof, threshold, horizon_hours=3),
        "constraint_contract": "conditional hazards are converted using 1 - cumulative_product(1 - hazard); no post-hoc monotonic projection",
        "censoring_contract": "only fully observed intervals are expanded; segment endpoints and internal timestamp gaps censor future intervals",
        "grouping_contract": "pre-onset rows share transition-event groups; stable controls share seven-day segment blocks",
        "severity_contract": "optional event-level weight is applied to every at-risk row of that event, not only the event interval",
    }


__all__ = [
    "HAZARD_HORIZONS",
    "HAZARD_INTERVALS",
    "TRANSITION_HAZARD_SCHEMA",
    "build_transition_hazard_labels",
    "causal_feature_columns",
    "cumulative_metrics",
    "event_recall_at_false_alerts",
    "expand_at_risk_rows",
    "fit_grouped_transition_hazard",
    "predict_cumulative_hazard",
]
