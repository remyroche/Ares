"""Causal base-model reliability context for the meta layer.

The base ledger contains OOS scores and eventually-resolved outcomes.  This
module converts that history into *as-of* features: every row on calendar day
``d`` sees only outcomes whose full label path ended before ``d``.  It is used
for meta training/replay and has no dependency on realised outcomes at the
row being scored.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


RELIABILITY_PREFIX = "base_reliability_"
RELIABILITY_FEATURE_COLUMNS = (
    "base_reliability_expected_soft_prior",
    "base_reliability_expected_hit_prior",
    "base_reliability_expected_ev_prior",
    "base_reliability_soft_calibration_error_prior",
    "base_reliability_hit_calibration_error_prior",
    "base_reliability_score_ic_soft_ewm_3d",
    "base_reliability_score_ic_ev_ewm_3d",
    "base_reliability_hr_surprise_ewm_3d",
    "base_reliability_ev_surprise_ewm_3d",
    "base_reliability_score_ic_soft_ewm_7d",
    "base_reliability_score_ic_ev_ewm_7d",
    "base_reliability_hr_surprise_ewm_7d",
    "base_reliability_ev_surprise_ewm_7d",
    "base_reliability_score_ic_soft_ewm_14d",
    "base_reliability_score_ic_ev_ewm_14d",
    "base_reliability_hr_surprise_ewm_14d",
    "base_reliability_ev_surprise_ewm_14d",
    "base_reliability_resolved_rows_prior",
    "base_reliability_resolved_days_prior",
    "base_reliability_local_support_weight",
)


def reliability_feature_columns() -> list[str]:
    return list(RELIABILITY_FEATURE_COLUMNS)


def _as_float(values: pd.Series | np.ndarray, fill: float = 0.0) -> np.ndarray:
    out = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    return np.nan_to_num(out, nan=fill, posinf=fill, neginf=fill)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 12:
        return 0.0
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    sx = float(xr.std())
    sy = float(yr.std())
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


@dataclass
class _ScopeState:
    """Small expanding calibration/reliability state for one scope."""

    band_count: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float64))
    band_soft: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float64))
    band_hit: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float64))
    band_ev: np.ndarray = field(default_factory=lambda: np.zeros(10, dtype=np.float64))
    rows: int = 0
    days: int = 0
    ewm: dict[int, np.ndarray] = field(
        default_factory=lambda: {half_life: np.zeros(4, dtype=np.float64) for half_life in (3, 7, 14)}
    )
    ewm_ready: dict[int, bool] = field(
        default_factory=lambda: {half_life: False for half_life in (3, 7, 14)}
    )

    def expected(self, band: np.ndarray, defaults: tuple[float, float, float]) -> np.ndarray:
        count = self.band_count[band]
        result = np.empty((len(band), 3), dtype=np.float64)
        for idx, sums in enumerate((self.band_soft, self.band_hit, self.band_ev)):
            result[:, idx] = np.divide(
                sums[band], count, out=np.full(len(band), defaults[idx], dtype=np.float64), where=count > 0
            )
        return result

    def update(self, band: np.ndarray, soft: np.ndarray, hit: np.ndarray, ev: np.ndarray, score: np.ndarray, expected: np.ndarray) -> None:
        counts = np.bincount(band, minlength=10).astype(np.float64)
        self.band_count += counts
        self.band_soft += np.bincount(band, weights=soft, minlength=10)
        self.band_hit += np.bincount(band, weights=hit, minlength=10)
        self.band_ev += np.bincount(band, weights=ev, minlength=10)
        self.rows += int(len(band))
        self.days += 1
        # Daily residuals are calculated against the pre-update calibration.
        metrics = np.array(
            [
                _safe_corr(score, soft),
                _safe_corr(score, ev),
                float(np.mean(hit - expected[:, 1])),
                float(np.mean(ev - expected[:, 2])),
            ],
            dtype=np.float64,
        )
        for half_life, prior in self.ewm.items():
            alpha = 1.0 - np.exp(-np.log(2.0) / float(half_life))
            self.ewm[half_life] = metrics if not self.ewm_ready[half_life] else (1.0 - alpha) * prior + alpha * metrics
            self.ewm_ready[half_life] = True


def _blend_expected(
    local: _ScopeState,
    side: _ScopeState,
    global_state: _ScopeState,
    band: np.ndarray,
    defaults: tuple[float, float, float],
) -> tuple[np.ndarray, float]:
    local_value = local.expected(band, defaults)
    side_value = side.expected(band, defaults)
    global_value = global_state.expected(band, defaults)
    # Empirical Bayes hierarchy: sparse side/archetype curves shrink to side,
    # then to global.  Counts are prior resolved rows only.
    local_n = local.band_count[band]
    side_n = side.band_count[band]
    local_weight = local_n / (local_n + 400.0)
    side_weight = side_n / (side_n + 1_500.0)
    blended = local_weight[:, None] * local_value + (1.0 - local_weight[:, None]) * (
        side_weight[:, None] * side_value + (1.0 - side_weight[:, None]) * global_value
    )
    return blended, float(np.mean(local_weight)) if len(local_weight) else 0.0


def _copy_ewm(output: dict[str, np.ndarray], positions: np.ndarray, local: _ScopeState, side: _ScopeState, global_state: _ScopeState) -> None:
    local_weight = min(1.0, local.rows / 1_500.0)
    side_weight = min(1.0, side.rows / 5_000.0)
    for half_life in (3, 7, 14):
        values = local_weight * local.ewm[half_life] + (1.0 - local_weight) * (
            side_weight * side.ewm[half_life] + (1.0 - side_weight) * global_state.ewm[half_life]
        )
        output[f"{RELIABILITY_PREFIX}score_ic_soft_ewm_{half_life}d"][positions] = values[0]
        output[f"{RELIABILITY_PREFIX}score_ic_ev_ewm_{half_life}d"][positions] = values[1]
        output[f"{RELIABILITY_PREFIX}hr_surprise_ewm_{half_life}d"][positions] = values[2]
        output[f"{RELIABILITY_PREFIX}ev_surprise_ewm_{half_life}d"][positions] = values[3]


class CausalBaseReliabilityBuilder:
    """Streaming version of :func:`derive_causal_base_reliability`.

    The full top-30 history can exceed the memory budget once pandas copies
    categorical keys.  This builder processes one UTC signal day at a time;
    only still-unresolved outcome cohorts remain in memory.
    """

    def __init__(self, defaults: tuple[float, float, float]) -> None:
        self.defaults = tuple(map(float, defaults))
        self.global_state = _ScopeState()
        self.side_states: defaultdict[str, _ScopeState] = defaultdict(_ScopeState)
        self.group_states: defaultdict[str, _ScopeState] = defaultdict(_ScopeState)
        self.pending: defaultdict[pd.Timestamp, list[pd.DataFrame]] = defaultdict(list)

    def _update_arrivals(self, day: pd.Timestamp) -> None:
        pending = self.pending.pop(pd.Timestamp(day), [])
        if not pending:
            return
        arriving = pd.concat(pending, ignore_index=True, copy=False)
        scopes = (
            ("__all__", arriving),
            *[(str(side), group) for side, group in arriving.groupby("__side__", observed=True, sort=False)],
            *[(str(group_key), group) for group_key, group in arriving.groupby("__group__", observed=True, sort=False)],
        )
        for key, cohort in scopes:
            state = self.global_state if key == "__all__" else (self.side_states[key] if "__" not in key else self.group_states[key])
            band = cohort["__band__"].to_numpy(dtype=np.int8)
            expected = state.expected(band, self.defaults)
            state.update(
                band,
                cohort["__soft__"].to_numpy(dtype=np.float64),
                cohort["__hit__"].to_numpy(dtype=np.float64),
                cohort["__ev__"].to_numpy(dtype=np.float64),
                cohort["__score__"].to_numpy(dtype=np.float64),
                expected,
            )

    def transform_day(
        self,
        frame: pd.DataFrame,
        *,
        timestamp_column: str = "__ts__",
        resolution_column: str = "__label_path_end_ts__",
        score_column: str = "score",
        rank_column: str = "base_rank_pct_by_timestamp_side",
        side_column: str = "side_name",
        archetype_column: str = "__archetype_policy_key__",
        soft_column: str = "__first_touch_policy_soft__",
        hit_column: str = "__first_touch_hit__",
        ev_column: str = "__first_touch_capture_net__",
    ) -> pd.DataFrame:
        if frame.empty:
            return frame.loc[:, [timestamp_column, "__symbol__", side_column, archetype_column]].copy()
        work = frame.copy(deep=False)
        work[timestamp_column] = pd.to_datetime(work[timestamp_column], utc=True, errors="coerce")
        work[resolution_column] = pd.to_datetime(work[resolution_column], utc=True, errors="coerce")
        if work[timestamp_column].isna().any() or work[resolution_column].isna().any():
            raise ValueError("Base reliability requires finite UTC timestamp and label-path-end values")
        days = work[timestamp_column].dt.normalize()
        if days.nunique() != 1:
            raise ValueError("CausalBaseReliabilityBuilder.transform_day expects one UTC signal day")
        day = pd.Timestamp(days.iloc[0])
        self._update_arrivals(day)
        side = work[side_column].astype(str).str.lower()
        archetype = work[archetype_column].astype(str).replace("", "unknown")
        group = side + "__" + archetype
        band = np.clip(np.floor(_as_float(work[rank_column], fill=0.5) * 10.0).astype(np.int8), 0, 9)
        result = work[[timestamp_column, "__symbol__", side_column, archetype_column]].copy()
        output = {column: np.zeros(len(work), dtype=np.float32) for column in RELIABILITY_FEATURE_COLUMNS}
        for group_key, positions in pd.Series(np.arange(len(work)), index=group).groupby(level=0, sort=False):
            pos = positions.to_numpy(dtype=np.int64)
            side_key = str(side.iloc[pos[0]])
            local = self.group_states[str(group_key)]
            side_state = self.side_states[side_key]
            expected, support_weight = _blend_expected(local, side_state, self.global_state, band[pos], self.defaults)
            output[f"{RELIABILITY_PREFIX}expected_soft_prior"][pos] = expected[:, 0]
            output[f"{RELIABILITY_PREFIX}expected_hit_prior"][pos] = expected[:, 1]
            output[f"{RELIABILITY_PREFIX}expected_ev_prior"][pos] = expected[:, 2]
            score = _as_float(work.iloc[pos][score_column])
            output[f"{RELIABILITY_PREFIX}soft_calibration_error_prior"][pos] = expected[:, 0] - score
            output[f"{RELIABILITY_PREFIX}hit_calibration_error_prior"][pos] = expected[:, 1] - score
            output[f"{RELIABILITY_PREFIX}resolved_rows_prior"][pos] = local.rows
            output[f"{RELIABILITY_PREFIX}resolved_days_prior"][pos] = local.days
            output[f"{RELIABILITY_PREFIX}local_support_weight"][pos] = support_weight
            _copy_ewm(output, pos, local, side_state, self.global_state)
        for column, values in output.items():
            result[column] = values

        queued = pd.DataFrame({
            "__available_day__": work[resolution_column].dt.normalize() + pd.Timedelta(days=1),
            "__side__": side.to_numpy(),
            "__group__": group.to_numpy(),
            "__band__": band,
            "__score__": _as_float(work[score_column]),
            "__soft__": _as_float(work[soft_column]),
            "__hit__": _as_float(work[hit_column]),
            "__ev__": _as_float(work[ev_column]),
        })
        for available_day, cohort in queued.groupby("__available_day__", observed=True, sort=False):
            self.pending[pd.Timestamp(available_day)].append(cohort.drop(columns="__available_day__").reset_index(drop=True))
        return result


def derive_causal_base_reliability(
    frame: pd.DataFrame,
    *,
    timestamp_column: str = "__ts__",
    resolution_column: str = "__label_path_end_ts__",
    score_column: str = "score",
    rank_column: str = "base_rank_pct_by_timestamp_side",
    side_column: str = "side_name",
    archetype_column: str = "__archetype_policy_key__",
    soft_column: str = "__first_touch_policy_soft__",
    hit_column: str = "__first_touch_hit__",
    ev_column: str = "__first_touch_capture_net__",
) -> pd.DataFrame:
    """Return keyed, causally available reliability features.

    A resolved row is admitted at the *next UTC day* after its label path end.
    Thus every signal on day D uses a full-calendar-day embargo, even when a
    short path happened to resolve intraday on D.
    """
    required = (timestamp_column, resolution_column, score_column, rank_column, side_column, archetype_column, soft_column, hit_column, ev_column)
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"Base reliability input is missing {missing}")
    work = frame.loc[:, list(dict.fromkeys([timestamp_column, "__symbol__", side_column, archetype_column, resolution_column, score_column, rank_column, soft_column, hit_column, ev_column]))].copy()
    work[timestamp_column] = pd.to_datetime(work[timestamp_column], utc=True, errors="coerce")
    work[resolution_column] = pd.to_datetime(work[resolution_column], utc=True, errors="coerce")
    if work[timestamp_column].isna().any() or work[resolution_column].isna().any():
        raise ValueError("Base reliability requires finite UTC timestamp and label-path-end values")
    work[side_column] = work[side_column].astype(str).str.lower()
    work[archetype_column] = work[archetype_column].astype(str).replace("", "unknown")
    work["__group__"] = work[side_column] + "__" + work[archetype_column]
    work["__signal_day__"] = work[timestamp_column].dt.normalize()
    work["__available_day__"] = work[resolution_column].dt.normalize() + pd.Timedelta(days=1)
    work["__score__"] = _as_float(work[score_column])
    rank = _as_float(work[rank_column], fill=0.5)
    work["__band__"] = np.clip(np.floor(rank * 10.0).astype(np.int8), 0, 9)
    work["__soft__"] = _as_float(work[soft_column])
    work["__hit__"] = _as_float(work[hit_column])
    work["__ev__"] = _as_float(work[ev_column])
    work = work.sort_values(["__signal_day__", timestamp_column, "__symbol__", side_column], kind="stable").reset_index(drop=True)

    n = len(work)
    output = {column: np.zeros(n, dtype=np.float32) for column in RELIABILITY_FEATURE_COLUMNS}
    global_state = _ScopeState()
    side_states: defaultdict[str, _ScopeState] = defaultdict(_ScopeState)
    group_states: defaultdict[str, _ScopeState] = defaultdict(_ScopeState)
    default_soft = float(np.clip(work["__soft__"].mean(), 0.0, 1.0))
    default_hit = float(np.clip(work["__hit__"].mean(), 0.0, 1.0))
    default_ev = float(work["__ev__"].mean())
    defaults = (default_soft, default_hit, default_ev)

    # Dict avoids a large list of empty calendar-date frames; arrival groups
    # are processed once, at their conservative availability date.
    arrivals: dict[pd.Timestamp, np.ndarray] = {
        day: index.to_numpy(dtype=np.int64)
        for day, index in work.groupby("__available_day__", sort=True).groups.items()
    }
    signal_days = work["__signal_day__"].to_numpy()
    unique_days = np.unique(signal_days)
    for day in unique_days:
        arrival = arrivals.get(pd.Timestamp(day))
        if arrival is not None:
            arriving = work.iloc[arrival]
            scopes = (
                ("__all__", arriving),
                *[(str(side), group) for side, group in arriving.groupby(side_column, observed=True, sort=False)],
                *[(str(group_key), group) for group_key, group in arriving.groupby("__group__", observed=True, sort=False)],
            )
            for key, cohort in scopes:
                state = global_state if key == "__all__" else (side_states[key] if "__" not in key else group_states[key])
                band = cohort["__band__"].to_numpy(dtype=np.int8)
                expected = state.expected(band, defaults)
                state.update(
                    band,
                    cohort["__soft__"].to_numpy(dtype=np.float64),
                    cohort["__hit__"].to_numpy(dtype=np.float64),
                    cohort["__ev__"].to_numpy(dtype=np.float64),
                    cohort["__score__"].to_numpy(dtype=np.float64),
                    expected,
                )
        positions = np.flatnonzero(signal_days == day)
        day_rows = work.iloc[positions]
        for group_key, cohort in day_rows.groupby("__group__", observed=True, sort=False):
            # ``work`` was reset to a dense index after sorting, so the cohort
            # index itself is the output position.  Indexing it through the
            # day-local ``positions`` array would incorrectly treat global row
            # IDs as offsets on all but the first day.
            pos = cohort.index.to_numpy(dtype=np.int64)
            side = str(cohort[side_column].iloc[0])
            local = group_states[str(group_key)]
            side_state = side_states[side]
            band = cohort["__band__"].to_numpy(dtype=np.int8)
            expected, support_weight = _blend_expected(local, side_state, global_state, band, defaults)
            output[f"{RELIABILITY_PREFIX}expected_soft_prior"][pos] = expected[:, 0]
            output[f"{RELIABILITY_PREFIX}expected_hit_prior"][pos] = expected[:, 1]
            output[f"{RELIABILITY_PREFIX}expected_ev_prior"][pos] = expected[:, 2]
            output[f"{RELIABILITY_PREFIX}soft_calibration_error_prior"][pos] = expected[:, 0] - cohort["__score__"].to_numpy(dtype=np.float64)
            output[f"{RELIABILITY_PREFIX}hit_calibration_error_prior"][pos] = expected[:, 1] - cohort["__score__"].to_numpy(dtype=np.float64)
            output[f"{RELIABILITY_PREFIX}resolved_rows_prior"][pos] = local.rows
            output[f"{RELIABILITY_PREFIX}resolved_days_prior"][pos] = local.days
            output[f"{RELIABILITY_PREFIX}local_support_weight"][pos] = support_weight
            _copy_ewm(output, pos, local, side_state, global_state)

    result = work[[timestamp_column, "__symbol__", side_column, archetype_column]].copy()
    for column, values in output.items():
        result[column] = values
    return result
