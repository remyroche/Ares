"""Fixed, low-retraining strict-OOF calendar for canonical 31/8 base scores.

The contract intentionally operates on pre-entry identity/time/feature
availability only.  It never accepts targets, realised returns, transition
labels, event IDs, or outcome-based candidate sampling as inputs.  Economic and
transition subsets are applied only after every frozen identity has been
scored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

import numpy as np
import pandas as pd


IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
REQUIRED = (*IDENTITY, "__decision_ts__", "__label_resolution_ts__")
SIDES = ("long", "short")


@dataclass(frozen=True)
class HistoricalOOFCalendarSpec:
    """Calendar parameters for the canonical base reconstruction only."""

    score_start: pd.Timestamp
    score_end: pd.Timestamp
    validation_frequency: str = "MS"
    label_resolution_hours: float = 24.0
    embargo_hours: float = 24.0
    minimum_train_rows_per_side: int = 50_000
    maximum_fit_rows_per_side: int = 100_000
    sampling_namespace: str = "canonical_31_8_historical_base_oof_v1"

    def __post_init__(self) -> None:
        start = _utc_timestamp(self.score_start, name="score_start")
        end = _utc_timestamp(self.score_end, name="score_end")
        if start >= end:
            raise ValueError("score_start must precede score_end")
        if start != start.normalize() or end != end.normalize():
            raise ValueError("monthly OOF boundaries must be UTC midnight")
        if self.validation_frequency != "MS":
            raise ValueError("only monthly-start validation blocks are supported")
        if self.label_resolution_hours <= 0 or self.embargo_hours < 0:
            raise ValueError("label resolution must be positive and embargo non-negative")
        if self.minimum_train_rows_per_side < 1 or self.maximum_fit_rows_per_side < 1:
            raise ValueError("training row limits must be positive")
        if self.maximum_fit_rows_per_side < self.minimum_train_rows_per_side:
            raise ValueError("maximum fit rows must cover the minimum per-side support")


def _utc_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    return timestamp.tz_convert("UTC")


def _utc(values: pd.Series, *, name: str) -> pd.Series:
    result = pd.to_datetime(values, utc=True, errors="coerce")
    if result.isna().any():
        raise ValueError(f"{name} contains invalid timestamps")
    return result


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    def safe(value: Any) -> Any:
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        return value

    return sha256(
        json.dumps(safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_identity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise an identity ledger and reject non-PIT/ambiguous candidates."""

    missing = sorted(set(REQUIRED).difference(frame.columns))
    if missing:
        raise ValueError(f"identity ledger is missing columns: {missing}")
    result = frame.loc[:, list(REQUIRED)].copy()
    for column in ("__ts__", "__decision_ts__", "__label_resolution_ts__"):
        result[column] = _utc(result[column], name=column)
    if result[["__symbol__", "candidate_id"]].isna().any().any():
        raise ValueError("identity ledger has incomplete candidate identity")
    result["__symbol__"] = result["__symbol__"].astype(str)
    result["side_name"] = result["side_name"].astype(str).str.lower()
    result["candidate_id"] = result["candidate_id"].astype(str)
    if not result["side_name"].isin(SIDES).all():
        raise ValueError("identity ledger must use canonical long/short side names")
    if result["candidate_id"].str.strip().eq("").any():
        raise ValueError("identity ledger has incomplete candidate identity")
    if result.duplicated(list(IDENTITY)).any() or result["candidate_id"].duplicated().any():
        raise ValueError("identity ledger has duplicate frozen candidate identities")
    if not result["__decision_ts__"].eq(result["__ts__"] + pd.Timedelta(hours=1)).all():
        raise ValueError("decision contract must be signal timestamp + 1h")
    if (result["__label_resolution_ts__"] < result["__decision_ts__"]).any():
        raise ValueError("label resolution cannot precede decision time")
    return result.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)


def _stable_rank(frame: pd.DataFrame, *, fold: str, side: str, namespace: str) -> pd.Series:
    payload = (
        namespace
        + "|"
        + fold
        + "|"
        + side
        + "|"
        + frame["candidate_id"].astype(str)
    )
    return payload.map(lambda value: sha256(value.encode("utf-8")).hexdigest())


def deterministic_calendar_sample(
    train: pd.DataFrame,
    *,
    fold: str,
    side: str,
    cap: int,
    namespace: str,
) -> pd.DataFrame:
    """Select a calendar-stratified, identity-hash sample without outcomes.

    Each complete UTC signal month receives a proportional quota.  Within the
    month, candidate-ID hashing is the sole tie-breaker.  Neither labels nor
    any future-derived event identifier can affect membership.
    """

    if len(train) <= cap:
        return train.sort_values(list(IDENTITY), kind="stable").copy()
    work = train.copy()
    # Conversion via a timezone-naive *copy* avoids the pandas warning while
    # retaining the UTC calendar semantics of the original timestamp.
    work["__calendar_month__"] = work["__ts__"].dt.tz_localize(None).dt.to_period("M").astype(str)
    counts = work.groupby("__calendar_month__", sort=True, observed=True).size()
    raw = counts.astype(float) / float(counts.sum()) * int(cap)
    quota = np.floor(raw).astype(int)
    remainder = int(cap - quota.sum())
    if remainder:
        fractions = (raw - quota).sort_values(ascending=False, kind="stable")
        for month in fractions.index[:remainder]:
            quota.loc[month] += 1
    # A proportional allocation can assign zero to a tiny month.  That is
    # intentional when it is too small for the fixed budget; recordable hash
    # sampling rather than event selection determines the omission.
    selected: list[pd.DataFrame] = []
    for month, local in work.groupby("__calendar_month__", sort=True, observed=True):
        take = int(quota.loc[str(month)])
        if take <= 0:
            continue
        ranked = local.assign(
            __sample_rank__=_stable_rank(local, fold=fold, side=side, namespace=namespace)
        ).sort_values(["__sample_rank__", "candidate_id"], kind="stable")
        selected.append(ranked.head(take).drop(columns=["__sample_rank__"]))
    result = pd.concat(selected, ignore_index=True) if selected else work.iloc[0:0].copy()
    return result.drop(columns=["__calendar_month__"], errors="ignore").sort_values(
        list(IDENTITY), kind="stable"
    ).reset_index(drop=True)


def _calendar_boundaries(spec: HistoricalOOFCalendarSpec) -> list[pd.Timestamp]:
    start = _utc_timestamp(spec.score_start, name="score_start")
    end = _utc_timestamp(spec.score_end, name="score_end")
    boundaries = list(pd.date_range(start, end, freq=spec.validation_frequency, tz="UTC"))
    if boundaries[-1] != end:
        boundaries.append(end)
    return boundaries


def build_historical_base_oof_calendar(
    identities: pd.DataFrame,
    *,
    spec: HistoricalOOFCalendarSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create exact per-side monthly OOF validation and fit-identity ledgers.

    Training is expanding and uses all feature-complete prior identities before
    applying a deterministic compute cap.  A transition/event window never
    enters this function, which is the key guard against outcome-selected
    reconstruction.
    """

    source = validate_identity_frame(identities)
    score_start = _utc_timestamp(spec.score_start, name="score_start")
    score_end = _utc_timestamp(spec.score_end, name="score_end")
    frozen = source.loc[source["__ts__"].ge(score_start) & source["__ts__"].lt(score_end)].copy()
    if frozen.empty:
        raise ValueError("no frozen score identities lie in the requested calendar")
    boundaries = _calendar_boundaries(spec)
    assignments: list[pd.DataFrame] = []
    train_samples: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    embargo = pd.Timedelta(hours=float(spec.embargo_hours))
    for number, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        fold = f"historical_base_{start.strftime('%Y%m')}"
        validation = frozen.loc[frozen["__ts__"].ge(start) & frozen["__ts__"].lt(end)].copy()
        if validation.empty:
            raise ValueError(f"{fold} has no frozen validation identities")
        for side in SIDES:
            valid_side = validation.loc[validation["side_name"].eq(side)].copy()
            # Both conditions are retained even though the canonical 24h
            # horizon makes them nearly equivalent: the first is outcome
            # resolution, the second is a documented dependency embargo.
            eligible = source.loc[
                source["side_name"].eq(side)
                & source["__label_resolution_ts__"].lt(start)
                & source["__decision_ts__"].lt(start - embargo)
            ].copy()
            if valid_side.empty:
                raise ValueError(f"{fold}/{side} has no frozen validation identities")
            if len(eligible) < int(spec.minimum_train_rows_per_side):
                raise ValueError(
                    f"{fold}/{side} has only {len(eligible)} resolved+embargoed training rows; "
                    f"requires {spec.minimum_train_rows_per_side}"
                )
            sample = deterministic_calendar_sample(
                eligible,
                fold=fold,
                side=side,
                cap=int(spec.maximum_fit_rows_per_side),
                namespace=spec.sampling_namespace,
            )
            if sample["__label_resolution_ts__"].ge(start).any() or sample["__decision_ts__"].ge(start - embargo).any():
                raise AssertionError("selected training sample violates resolution/purge contract")
            if set(sample["candidate_id"]).intersection(valid_side["candidate_id"]):
                raise AssertionError("a frozen validation identity entered its own training fold")
            valid_side["oof_fold"] = fold
            valid_side["validation_start_utc"] = start
            valid_side["validation_end_utc"] = end
            assignments.append(valid_side)
            selected = sample.copy()
            selected["oof_fold"] = fold
            selected["validation_start_utc"] = start
            selected["validation_end_utc"] = end
            train_samples.append(selected)
            fold_records.append(
                {
                    "fold": fold,
                    "side": side,
                    "validation_start_utc": start,
                    "validation_end_utc": end,
                    "validation_rows": int(len(valid_side)),
                    "training_rows_resolved_and_embargoed": int(len(eligible)),
                    "fit_rows": int(len(sample)),
                    "training_identity_sha256": canonical_json_sha256(
                        {"candidate_ids": sorted(sample["candidate_id"].astype(str).tolist())}
                    ),
                    "validation_identity_sha256": canonical_json_sha256(
                        {"candidate_ids": sorted(valid_side["candidate_id"].astype(str).tolist())}
                    ),
                    "max_training_decision_utc": sample["__decision_ts__"].max(),
                    "max_training_label_resolution_utc": sample["__label_resolution_ts__"].max(),
                }
            )
    frozen_assignments = pd.concat(assignments, ignore_index=True)
    if frozen_assignments["candidate_id"].duplicated().any() or len(frozen_assignments) != len(frozen):
        raise AssertionError("every frozen identity must receive exactly one strict OOF fold")
    sampled_train = pd.concat(train_samples, ignore_index=True)
    contract = {
        "schema": "canonical_31_8_historical_base_oof_calendar_v1",
        "scope": "base model only; a residual/meta model must consume earlier base OOF scores under its own purged calendar",
        "calendar": {
            "score_start_utc": score_start,
            "score_end_utc_exclusive": score_end,
            "validation_frequency": spec.validation_frequency,
            "folds": fold_records,
        },
        "compute_plan": {
            "base_model_fits": int(len(boundaries) - 1) * len(SIDES),
            "fit_granularity": "one expanding per-side base fit per UTC calendar-month validation block",
            "not_per_event": True,
            "valid_shortcut": "deterministically cap only the pre-fold all-candidate training population",
            "invalid_shortcuts": [
                "one all-February-April fit used to score February-April",
                "prioritising ex-post transition/event windows in training or scoring",
                "using returns, labels, or evaluation outcomes to select fit rows",
            ],
        },
        "time_contract": {
            "signal_to_decision": "__decision_ts__ = __ts__ + 1h",
            "label_resolution": "__label_resolution_ts__ = __decision_ts__ + 24h",
            "train_rule": "__label_resolution_ts__ < validation_start AND __decision_ts__ < validation_start - 24h",
            "validation_rule": "score every frozen feature-complete identity with validation_start <= __ts__ < validation_end",
            "purge_embargo_hours": float(spec.embargo_hours),
        },
        "side_contract": "long and short models, train samples, feature contracts, and predictions are strictly separate",
        "sampling_contract": {
            "training_population": "all prior resolved+embargoed feature-complete identities for that side",
            "compute_cap_per_side": int(spec.maximum_fit_rows_per_side),
            "method": "proportional UTC-calendar-month quota then stable SHA-256(candidate_id) rank",
            "forbidden_inputs": ["targets", "returns", "labels", "transition_event_id", "economic_failure", "evaluation_outcomes"],
        },
        "transition_window_policy": {
            "training": "FORBIDDEN: canonical/ex-post transition windows may not alter train membership, fold boundaries, feature selection, HPO, or sampling",
            "scoring": "FORBIDDEN: all frozen identities must be scored before any transition subset is joined",
            "reporting": "ALLOWED only as a labelled ex-post diagnostic subset; it is not independent promotion evidence",
        },
        "frozen_score_identity_count": int(len(frozen_assignments)),
        "fit_identity_count": int(len(sampled_train)),
    }
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return frozen_assignments.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True), sampled_train.sort_values(
        ["oof_fold", *IDENTITY], kind="stable"
    ).reset_index(drop=True), contract


def validate_strict_oof_predictions(
    frozen_assignments: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """Verify a later trainer scored every frozen identity once in its fold."""

    required = [*IDENTITY, "oof_fold"]
    missing = sorted(set(required).difference(predictions.columns))
    if missing:
        raise ValueError(f"predictions lack OOF identity columns: {missing}")
    expected = frozen_assignments.loc[:, required].copy()
    actual = predictions.loc[:, required].copy()
    if actual.duplicated(list(IDENTITY)).any() or len(actual) != len(expected):
        raise ValueError("predictions do not have exactly one row per frozen identity")
    joined = expected.merge(actual, on=list(IDENTITY), how="outer", suffixes=("__expected", "__actual"), indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("predictions differ from the frozen identity ledger")
    if not joined["oof_fold__expected"].eq(joined["oof_fold__actual"]).all():
        raise ValueError("a prediction was generated by the wrong OOF fold")
