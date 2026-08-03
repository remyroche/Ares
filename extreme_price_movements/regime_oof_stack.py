"""Contracts for candidate-keyed, leakage-safe OOF regime outputs.

This module deliberately contains no regime model.  It is the shared boundary
between a fold-local regime materializer and base/residual/calibration runners:
all outputs are candidate-keyed, carry their temporal provenance, and are
validated before they can be joined to a trading population.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


IDENTITY_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "__ts__",
    "__symbol__",
    "side_name",
)
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "regime_fold_id",
    "regime_train_end_utc",
    "regime_available_utc",
)
STATE_PROBABILITY_PREFIX = "regime_state_p__"
STATE_ID_COLUMN = "regime_state_id"
STATE_ENTROPY_COLUMN = "regime_state_entropy"
STATE_MARGIN_COLUMN = "regime_state_margin"
STATE_OOD_COLUMN = "regime_state_ood_score"
STATE_UNCERTAINTY_COLUMN = "regime_state_uncertainty"
TRANSITION_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "transition_fold_id",
    "transition_train_end_utc",
    "transition_available_utc",
)
TRANSITION_PROBABILITY_PREFIX = "transition_state_p__"
TRANSITION_ID_COLUMN = "transition_state_id"
TRANSITION_ENTROPY_COLUMN = "transition_state_entropy"
TRANSITION_MARGIN_COLUMN = "transition_state_margin"
TRANSITION_OOD_COLUMN = "transition_state_ood_score"
TRANSITION_UNCERTAINTY_COLUMN = "transition_state_uncertainty"


@dataclass(frozen=True)
class SoftLayerSpec:
    """Names and provenance fields for one independently learned soft layer."""

    name: str
    probability_prefix: str
    identity_column: str
    entropy_column: str
    margin_column: str
    ood_column: str
    uncertainty_column: str
    provenance_columns: tuple[str, ...]


REGIME_STATE_LAYER = SoftLayerSpec(
    name="regime_state",
    probability_prefix=STATE_PROBABILITY_PREFIX,
    identity_column=STATE_ID_COLUMN,
    entropy_column=STATE_ENTROPY_COLUMN,
    margin_column=STATE_MARGIN_COLUMN,
    ood_column=STATE_OOD_COLUMN,
    uncertainty_column=STATE_UNCERTAINTY_COLUMN,
    provenance_columns=PROVENANCE_COLUMNS,
)
TRANSITION_STATE_LAYER = SoftLayerSpec(
    name="transition_state",
    probability_prefix=TRANSITION_PROBABILITY_PREFIX,
    identity_column=TRANSITION_ID_COLUMN,
    entropy_column=TRANSITION_ENTROPY_COLUMN,
    margin_column=TRANSITION_MARGIN_COLUMN,
    ood_column=TRANSITION_OOD_COLUMN,
    uncertainty_column=TRANSITION_UNCERTAINTY_COLUMN,
    provenance_columns=TRANSITION_PROVENANCE_COLUMNS,
)

# Outcome and post-resolution columns cannot enter a regime OOS transform.
# This is deliberately conservative: callers may extend it, but cannot weaken
# the core contract by silently allowing common target aliases.
OUTCOME_DENYLIST_EXACT = frozenset(
    {
        "target_soft",
        "target_hard",
        "clean_exec",
        "clean_exec_label",
        "dirty_positive",
        "timeout",
        "exec_margin",
        "ev_after_1pct",
        "execution_net_ev_12h",
        "ret_net",
        "u_policy_net",
        "first_touch_net",
        "first_touch_gross",
        "full_path_bad_mae_1r",
        "first_touch_bad_mae_1r",
        "mfe_before_mae_1r",
        "mae_before_mfe_1r",
        "base_label_resolution_utc",
        "execution_label_end_utc",
        "outcomes_available",
    }
)
OUTCOME_DENYLIST_PREFIXES = (
    "target_",
    "label_",
    "expost_",
    "realized_",
    "outcome_",
    "future_",
    "post_entry_",
)


class RegimeOOFStackError(ValueError):
    """Raised when an OOF regime boundary is not safe to use."""


def _utc(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise RegimeOOFStackError(f"{name} contains non-UTC/invalid timestamps")
    return parsed


def _require(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise RegimeOOFStackError(f"{context} missing required columns: {missing}")


def assert_outcome_free(
    frame: pd.DataFrame, *, extra_forbidden: Iterable[str] = ()
) -> None:
    """Reject known realized-outcome fields from a regime feature frame."""

    extra = {str(name).lower() for name in extra_forbidden}
    leaked: list[str] = []
    for name in frame.columns:
        lower = str(name).lower()
        if (
            lower in OUTCOME_DENYLIST_EXACT
            or lower in extra
            or lower.startswith(OUTCOME_DENYLIST_PREFIXES)
        ):
            leaked.append(str(name))
    if leaked:
        raise RegimeOOFStackError(
            "regime OOS frame contains outcome-derived columns: "
            f"{sorted(leaked)[:12]}"
        )


def validate_candidate_identity(
    frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str] = IDENTITY_COLUMNS,
    require_unique_candidate_id: bool = True,
) -> pd.DataFrame:
    """Validate and UTC-normalize the immutable candidate identity."""

    _require(frame, identity_columns, "candidate frame")
    out = frame.copy()
    out["__ts__"] = _utc(out["__ts__"], "candidate __ts__")
    for column in ("candidate_id", "__symbol__", "side_name"):
        if out[column].isna().any() or out[column].astype(str).str.strip().eq("").any():
            raise RegimeOOFStackError(f"candidate identity column {column} has null/empty values")
    if out.duplicated(list(identity_columns)).any():
        raise RegimeOOFStackError("candidate identity is not unique")
    if require_unique_candidate_id and out["candidate_id"].duplicated().any():
        raise RegimeOOFStackError("candidate_id must be unique in a candidate population")
    return out


def probability_columns(
    frame: pd.DataFrame, *, prefix: str = STATE_PROBABILITY_PREFIX
) -> list[str]:
    """Return sorted, aligned state-simplex columns."""

    columns = sorted(str(column) for column in frame.columns if str(column).startswith(prefix))
    if not columns:
        raise RegimeOOFStackError(f"no aligned state probability columns with prefix {prefix!r}")
    suffixes = [column.removeprefix(prefix) for column in columns]
    if any(not suffix or suffix in {"unknown", "unmapped", "raw"} for suffix in suffixes):
        raise RegimeOOFStackError("state probability columns must use stable aligned state IDs")
    if len(set(suffixes)) != len(suffixes):
        raise RegimeOOFStackError("duplicate aligned state IDs")
    return columns


def derive_soft_state_fields(
    frame: pd.DataFrame,
    *,
    probability_prefix: str = STATE_PROBABILITY_PREFIX,
    layer: SoftLayerSpec | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    """Derive stable identity, entropy, margin and uncertainty from a simplex."""

    spec = layer or (
        REGIME_STATE_LAYER
        if probability_prefix == STATE_PROBABILITY_PREFIX
        else TRANSITION_STATE_LAYER
        if probability_prefix == TRANSITION_PROBABILITY_PREFIX
        else SoftLayerSpec(
            name="custom_state",
            probability_prefix=probability_prefix,
            identity_column=STATE_ID_COLUMN,
            entropy_column=STATE_ENTROPY_COLUMN,
            margin_column=STATE_MARGIN_COLUMN,
            ood_column=STATE_OOD_COLUMN,
            uncertainty_column=STATE_UNCERTAINTY_COLUMN,
            provenance_columns=(),
        )
    )
    out = frame.copy() if copy else frame
    columns = probability_columns(out, prefix=spec.probability_prefix)
    probabilities = out.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(probabilities).all() or (probabilities < -1e-8).any():
        raise RegimeOOFStackError("state probabilities must be finite and non-negative")
    sums = probabilities.sum(axis=1)
    if not np.allclose(sums, 1.0, rtol=0.0, atol=1e-6):
        raise RegimeOOFStackError("state probabilities must sum to one before deriving fields")
    order = np.argsort(probabilities, axis=1)
    winner = order[:, -1]
    max_probability = probabilities[np.arange(len(out)), winner]
    second = probabilities[np.arange(len(out)), order[:, -2]] if probabilities.shape[1] > 1 else np.zeros(len(out))
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)), axis=1)
    entropy /= np.log(float(max(2, probabilities.shape[1])))
    suffixes = [column.removeprefix(spec.probability_prefix) for column in columns]
    out[spec.identity_column] = np.asarray([suffixes[index] for index in winner], dtype=object)
    out[spec.entropy_column] = entropy.astype(np.float32)
    out[spec.margin_column] = (max_probability - second).astype(np.float32)
    out[spec.uncertainty_column] = (0.5 * entropy + 0.5 * (1.0 - max_probability)).astype(np.float32)
    return out


def _validate_soft_layer(
    frame: pd.DataFrame,
    *,
    layer: SoftLayerSpec,
    identity_columns: Sequence[str] = IDENTITY_COLUMNS,
    decision_timestamp_col: str = "__ts__",
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Validate one soft layer without letting it stand in for another layer."""

    assert_outcome_free(frame)
    out = validate_candidate_identity(frame, identity_columns=identity_columns)
    _require(out, layer.provenance_columns, f"{layer.name} output")
    _require(
        out,
        (
            layer.identity_column,
            layer.entropy_column,
            layer.margin_column,
            layer.ood_column,
            layer.uncertainty_column,
        ),
        f"{layer.name} output",
    )
    fold_col, train_end_col, available_col = layer.provenance_columns
    out[train_end_col] = _utc(out[train_end_col], train_end_col)
    out[available_col] = _utc(out[available_col], available_col)
    decision = _utc(out[decision_timestamp_col], decision_timestamp_col)
    if not out[train_end_col].lt(decision).all():
        raise RegimeOOFStackError(f"{train_end_col} must be strictly before every candidate decision")
    if not out[available_col].le(decision).all():
        raise RegimeOOFStackError(f"{available_col} must be at or before every candidate decision")
    if out[fold_col].isna().any() or out[fold_col].astype(str).str.strip().eq("").any():
        raise RegimeOOFStackError(f"{fold_col} must be populated")

    columns = probability_columns(out, prefix=layer.probability_prefix)
    probabilities = out.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(probabilities).all() or (probabilities < -tolerance).any():
        raise RegimeOOFStackError(f"{layer.name} probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=tolerance):
        raise RegimeOOFStackError(f"{layer.name} aligned probability simplex does not sum to one")
    suffixes = np.asarray([column.removeprefix(layer.probability_prefix) for column in columns], dtype=object)
    expected_identity = suffixes[np.argmax(probabilities, axis=1)].astype(str)
    observed_identity = out[layer.identity_column].astype(str).to_numpy()
    if not np.array_equal(expected_identity, observed_identity):
        raise RegimeOOFStackError(f"{layer.identity_column} must equal the aligned simplex argmax")
    for column in (layer.entropy_column, layer.margin_column, layer.uncertainty_column):
        values = pd.to_numeric(out[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (values < -tolerance).any() or (values > 1.0 + tolerance).any():
            raise RegimeOOFStackError(f"{column} must be finite and bounded in [0, 1]")
    ood = pd.to_numeric(out[layer.ood_column], errors="coerce").to_numpy(float)
    if not np.isfinite(ood).all() or (ood < -tolerance).any():
        raise RegimeOOFStackError(f"{layer.ood_column} must be finite and non-negative")
    return out


def validate_regime_output_frame(
    frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str] = IDENTITY_COLUMNS,
    decision_timestamp_col: str = "__ts__",
    probability_prefix: str = STATE_PROBABILITY_PREFIX,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Validate candidate-keyed soft regime outputs and temporal provenance."""

    if probability_prefix != STATE_PROBABILITY_PREFIX:
        raise RegimeOOFStackError("regime-state validation only accepts regime_state_p__ columns")
    return _validate_soft_layer(
        frame,
        layer=REGIME_STATE_LAYER,
        identity_columns=identity_columns,
        decision_timestamp_col=decision_timestamp_col,
        tolerance=tolerance,
    )


def validate_transition_output_frame(
    frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str] = IDENTITY_COLUMNS,
    decision_timestamp_col: str = "__ts__",
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Validate a transition layer; regime-state fields cannot substitute for it."""

    return _validate_soft_layer(
        frame,
        layer=TRANSITION_STATE_LAYER,
        identity_columns=identity_columns,
        decision_timestamp_col=decision_timestamp_col,
        tolerance=tolerance,
    )


def validate_combined_regime_transition_outputs(
    frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str] = IDENTITY_COLUMNS,
) -> pd.DataFrame:
    """Require both independently-provenanced state and transition layers."""

    regime = validate_regime_output_frame(frame, identity_columns=identity_columns)
    transition = validate_transition_output_frame(frame, identity_columns=identity_columns)
    if not regime.loc[:, list(identity_columns)].equals(transition.loc[:, list(identity_columns)]):
        raise RegimeOOFStackError("regime and transition layers do not share exact candidate identities")
    return frame.copy()


def exact_join_regime_outputs(
    candidates: pd.DataFrame,
    regime_outputs: pd.DataFrame,
    *,
    identity_columns: Sequence[str] = IDENTITY_COLUMNS,
) -> pd.DataFrame:
    """Exact one-to-one candidate join that preserves the input population."""

    left = validate_candidate_identity(candidates, identity_columns=identity_columns)
    right = validate_regime_output_frame(regime_outputs, identity_columns=identity_columns)
    value_columns = [column for column in right.columns if column not in identity_columns]
    collisions = sorted(set(value_columns).intersection(left.columns))
    if collisions:
        raise RegimeOOFStackError(f"candidate/regime feature collision: {collisions[:12]}")
    joined = left.merge(
        right.loc[:, [*identity_columns, *value_columns]],
        on=list(identity_columns),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(joined) != len(left) or joined["regime_fold_id"].isna().any():
        raise RegimeOOFStackError("exact regime join lost candidates or lacks OOF regime outputs")
    if joined.duplicated(list(identity_columns)).any():
        raise RegimeOOFStackError("exact regime join duplicated candidate identities")
    return joined


def exact_join_transition_outputs(
    candidates: pd.DataFrame,
    transition_outputs: pd.DataFrame,
    *,
    identity_columns: Sequence[str] = IDENTITY_COLUMNS,
) -> pd.DataFrame:
    """Exact one-to-one transition-layer join preserving all candidates."""

    left = validate_candidate_identity(candidates, identity_columns=identity_columns)
    right = validate_transition_output_frame(transition_outputs, identity_columns=identity_columns)
    value_columns = [column for column in right.columns if column not in identity_columns]
    collisions = sorted(set(value_columns).intersection(left.columns))
    if collisions:
        raise RegimeOOFStackError(f"candidate/transition feature collision: {collisions[:12]}")
    joined = left.merge(
        right.loc[:, [*identity_columns, *value_columns]],
        on=list(identity_columns),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(joined) != len(left) or joined["transition_fold_id"].isna().any():
        raise RegimeOOFStackError("exact transition join lost candidates or lacks OOF transition outputs")
    if joined.duplicated(list(identity_columns)).any():
        raise RegimeOOFStackError("exact transition join duplicated candidate identities")
    return joined


def combine_regime_transition_feature_view(
    candidates: pd.DataFrame,
    regime_outputs: pd.DataFrame,
    transition_outputs: pd.DataFrame,
    *,
    identity_columns: Sequence[str] = IDENTITY_COLUMNS,
) -> pd.DataFrame:
    """Build a matched feature view retaining both independent soft layers."""

    with_regime = exact_join_regime_outputs(
        candidates, regime_outputs, identity_columns=identity_columns
    )
    combined = exact_join_transition_outputs(
        with_regime, transition_outputs, identity_columns=identity_columns
    )
    return validate_combined_regime_transition_outputs(
        combined, identity_columns=identity_columns
    )


@dataclass(frozen=True)
class RegimeTransitionAblationArm:
    """Matched model arm definition; all arms use the same candidate population."""

    name: str
    include_regime_state: bool
    include_transition_state: bool


def matched_regime_transition_ablation_arms() -> tuple[RegimeTransitionAblationArm, ...]:
    """The only four arms for a clean regime-versus-transition comparison."""

    return (
        RegimeTransitionAblationArm("baseline", False, False),
        RegimeTransitionAblationArm("regime_only", True, False),
        RegimeTransitionAblationArm("transition_only", False, True),
        RegimeTransitionAblationArm("regime_plus_transition", True, True),
    )


def asof_join_regime_timeline(
    candidates: pd.DataFrame,
    timeline: pd.DataFrame,
    *,
    by: Sequence[str] = ("side_name",),
    candidate_timestamp_col: str = "__ts__",
    timeline_timestamp_col: str = "regime_source_utc",
    max_lag: pd.Timedelta | None = None,
    provenance_columns: Sequence[str] = PROVENANCE_COLUMNS,
) -> pd.DataFrame:
    """Backward as-of join a market-state timeline without row loss or look-ahead.

    The timeline intentionally has no candidate IDs.  It must be unique by
    ``by + regime_source_utc`` and its output availability is checked against
    each matched candidate after the join.
    """

    left = validate_candidate_identity(candidates)
    assert_outcome_free(timeline)
    _require(timeline, (*by, timeline_timestamp_col, *provenance_columns), "regime timeline")
    right = timeline.copy()
    left[candidate_timestamp_col] = _utc(left[candidate_timestamp_col], candidate_timestamp_col)
    right[timeline_timestamp_col] = _utc(right[timeline_timestamp_col], timeline_timestamp_col)
    if len(provenance_columns) != 3:
        raise RegimeOOFStackError("timeline provenance must be fold ID, train end and availability")
    _fold_col, train_end_col, available_col = provenance_columns
    right[train_end_col] = _utc(right[train_end_col], train_end_col)
    right[available_col] = _utc(right[available_col], available_col)
    if right.duplicated([*by, timeline_timestamp_col]).any():
        raise RegimeOOFStackError("regime timeline is not unique by group and source timestamp")
    forbidden = set(left.columns).intersection(set(right.columns) - set(by) - {timeline_timestamp_col})
    if forbidden:
        raise RegimeOOFStackError(f"candidate/timeline feature collision: {sorted(forbidden)[:12]}")
    left["__regime_join_order__"] = np.arange(len(left), dtype=np.int64)
    merge_kwargs: dict[str, Any] = {}
    if by:
        merge_kwargs["by"] = list(by)
    joined = pd.merge_asof(
        left.sort_values([*by, candidate_timestamp_col], kind="stable"),
        right.sort_values([*by, timeline_timestamp_col], kind="stable"),
        left_on=candidate_timestamp_col,
        right_on=timeline_timestamp_col,
        direction="backward",
        allow_exact_matches=True,
        tolerance=max_lag,
        **merge_kwargs,
    ).sort_values("__regime_join_order__", kind="stable")
    if len(joined) != len(left) or joined.duplicated(list(IDENTITY_COLUMNS)).any():
        raise RegimeOOFStackError("as-of regime join changed candidate identity cardinality")
    if joined[timeline_timestamp_col].isna().any():
        raise RegimeOOFStackError("as-of regime join has uncovered candidate rows")
    decision = _utc(joined[candidate_timestamp_col], candidate_timestamp_col)
    if not joined[train_end_col].lt(decision).all():
        raise RegimeOOFStackError("as-of timeline train end is not strictly before candidate decision")
    if not joined[available_col].le(decision).all():
        raise RegimeOOFStackError("as-of timeline output was unavailable at candidate decision")
    if not joined[timeline_timestamp_col].le(decision).all():
        raise RegimeOOFStackError("as-of regime source is after candidate decision")
    return joined.drop(columns="__regime_join_order__")


def utc_period_key(timestamp: pd.Series, period_type: str) -> pd.Series:
    """UTC Monday-start week or calendar-month key."""

    values = _utc(timestamp, "timestamp")
    if period_type == "week":
        naive = values.dt.tz_convert("UTC").dt.tz_localize(None)
        return naive.dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")
    if period_type == "month":
        return values.dt.strftime("%Y-%m")
    raise RegimeOOFStackError("period_type must be 'week' or 'month'")


def period_q10_q50(
    frame: pd.DataFrame,
    *,
    value_col: str,
    period_type: str,
    timestamp_col: str = "__ts__",
) -> dict[str, float | int | str]:
    """Return cross-period Q10/Q50 of period mean values.

    These are stability statistics over observed calendar periods, not
    bootstrap confidence limits.  The caller should pass the exact selected
    global-top-k rows for an economic metric, or all scored rows for IC only
    after computing one IC value per period externally.
    """

    _require(frame, (timestamp_col, value_col), "period metric frame")
    values = pd.to_numeric(frame[value_col], errors="coerce")
    work = pd.DataFrame({"period": utc_period_key(frame[timestamp_col], period_type), "value": values})
    means = work.groupby("period", observed=True)["value"].mean().dropna()
    return {
        "period_type": period_type,
        "periods": int(len(means)),
        "q10": float(means.quantile(0.10)) if len(means) else float("nan"),
        "q50": float(means.quantile(0.50)) if len(means) else float("nan"),
        "mean": float(means.mean()) if len(means) else float("nan"),
    }


def qualify_category_stability(
    frame: pd.DataFrame,
    *,
    category_col: str,
    value_col: str,
    timestamp_col: str = "__ts__",
    min_rows: int = 400,
    min_weeks: int = 12,
    min_months: int = 4,
    min_same_sign_week_fraction: float = 0.75,
    chronological_blocks: int = 4,
) -> pd.DataFrame:
    """Qualify categories whose economic delta is stable, not merely large.

    Each category is compared with the contemporaneous all-category mean.  The
    result is descriptive: promotion additionally requires fold-separated
    evidence and bootstrap support in the eventual evaluator.
    """

    _require(frame, (category_col, value_col, timestamp_col), "category stability frame")
    if not 0.0 < min_same_sign_week_fraction <= 1.0:
        raise RegimeOOFStackError("min_same_sign_week_fraction must be in (0, 1]")
    work = frame.loc[:, [category_col, value_col, timestamp_col]].copy()
    work["value"] = pd.to_numeric(work.pop(value_col), errors="coerce")
    work["week"] = utc_period_key(work[timestamp_col], "week")
    work["month"] = utc_period_key(work[timestamp_col], "month")
    work = work.dropna(subset=[category_col, "value"])
    baseline = work.groupby("week", observed=True)["value"].mean().rename("baseline")
    outputs: list[dict[str, Any]] = []
    for category, group in work.groupby(category_col, dropna=False, observed=True):
        weekly = group.groupby("week", observed=True)["value"].mean().rename("category_value").to_frame().join(baseline)
        weekly["delta"] = weekly["category_value"] - weekly["baseline"]
        delta = weekly["delta"].dropna().sort_index()
        mean_delta = float(delta.mean()) if len(delta) else float("nan")
        sign = float(np.sign(mean_delta)) if np.isfinite(mean_delta) else 0.0
        if sign == 0.0:
            same_sign = 0.0
            block_sign = 0.0
        else:
            same_sign = float((np.sign(delta) == sign).mean()) if len(delta) else 0.0
            blocks = [block for block in np.array_split(delta.to_numpy(float), max(1, int(chronological_blocks))) if len(block)]
            block_sign = float(np.mean([np.sign(np.mean(block)) == sign for block in blocks])) if blocks else 0.0
        rows = int(len(group))
        weeks = int(group["week"].nunique())
        months = int(group["month"].nunique())
        eligible = bool(
            rows >= int(min_rows)
            and weeks >= int(min_weeks)
            and months >= int(min_months)
            and sign != 0.0
            and same_sign >= float(min_same_sign_week_fraction)
            and block_sign >= float(min_same_sign_week_fraction)
        )
        outputs.append(
            {
                "category": str(category),
                "rows": rows,
                "weeks": weeks,
                "months": months,
                "mean_delta": mean_delta,
                "direction": "positive" if sign > 0 else "negative" if sign < 0 else "flat",
                "same_sign_week_fraction": same_sign,
                "same_sign_chronological_block_fraction": block_sign,
                "stable_category": eligible,
            }
        )
    return pd.DataFrame(outputs).sort_values("category", kind="stable").reset_index(drop=True)
