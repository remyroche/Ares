"""Strict, bounded H1--H5 health materialisation from contribution events.

This is the production replacement for the scope-wide SQL plans.  It consumes
only a bounded decision-month window from the immutable contribution-event
sidecar, scores complete equal-timestamp blocks before applying equal-time
labels, and retains dense state only for the current ``(contract, side,
head)`` scope.  Selected H3/H4/H5 evidence is the sole contribution-level
output that crosses the scope boundary.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .causal_leaf_covariance import (
    HORIZONS,
    CausalLeafCovarianceConfig,
    _EWMCovariance,
    _diagnostics,
    _hierarchical_matrices,
    covariance_feature_names,
)
from .causal_leaf_health import (
    DIRECTIONS,
    HEADS,
    SCHEMA,
    STATUS,
    CausalLeafHealthConfig,
    CausalLeafHealthError,
    _config_payload,
    _fit_ridge,
    _relationship_break_columns,
)
from .causal_leaf_health_incremental import (
    H1_METRIC_COUNT,
    allocate_h1_state,
    score_auxiliary_block,
    score_h1_block,
    update_h1_block,
)
from .causal_leaf_health_scoped import (
    _candidate_global_resolution,
    _copy_h5,
    _paths_literal,
    _pivot_partial,
    _scope_rows,
    _write_h4_cells,
)
from .causal_leaf_health_vectorized import _H1_METRICS, _H2_METRICS, _H3_METRICS, _IDENTITY
from .strict_contribution_event_reader import (
    ContributionEventGroup,
    ContributionEventTimestampBlock,
    iter_contribution_event_timestamp_blocks,
)
from .strict_contribution_event_stream import StrictContributionEventStreams, load_strict_contribution_event_streams
from .strict_event_store import CANDIDATE_COLUMNS, StrictEventStore, StrictEventStoreError, load_strict_event_store


_SCOPE = ("contract", "side", "head")
_FAMILY = ("feature_contract_sha256", "side_name", "head_name", "rule_signature", "contribution_direction")
_SELECTED_STATE_LIMIT_DEFAULT = 3_000_000
_DIRECT_WRITER_ROWS = 16_384


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _literal(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _utc_ns(value: object) -> int:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return int(stamp.value)


def _timestamp(value: int) -> pd.Timestamp:
    return pd.Timestamp(int(value), tz="UTC")


def _month_key(timestamp_ns: int) -> str:
    return _timestamp(timestamp_ns).strftime("%Y-%m")


def _month_end_ns(month: str) -> int:
    return int((pd.Timestamp(f"{month}-01", tz="UTC") + pd.offsets.MonthBegin(1)).value)


def _selection_active(timestamp_ns: int, config: CausalLeafHealthConfig) -> bool:
    if config.family_selection_effective_utc is None:
        return True
    return int(timestamp_ns) >= _utc_ns(config.family_selection_effective_utc)


def _context_arrays(context: pd.DataFrame, columns: Sequence[str], config: CausalLeafHealthConfig) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if "candidate_id" in context.columns:
        raise CausalLeafHealthError("event-incremental health requires a shared causal context timeline")
    selected = tuple(map(str, columns[: int(config.covariance_max_fields)]))
    required = {"regime_available_utc", *selected}
    missing = sorted(required.difference(context.columns))
    if missing:
        raise CausalLeafHealthError(f"causal context lacks declared fields: {missing}")
    forbidden = (
        "target", "label", "outcome", "future", "realized", "realised", "pnl", "net_ev",
        "gross_ev", "mfe", "mae", "barrier", "timeout", "exit", "post_entry", "postentry",
    )
    invalid = [str(name) for name in context.columns if any(token in str(name).lower() for token in forbidden)]
    if invalid:
        raise CausalLeafHealthError(f"causal context contains outcome-derived fields: {invalid[:8]}")
    work = context.loc[:, ["regime_available_utc", *selected]].copy()
    work["regime_available_utc"] = pd.to_datetime(work["regime_available_utc"], utc=True, errors="coerce")
    if work["regime_available_utc"].isna().any() or work["regime_available_utc"].duplicated().any():
        raise CausalLeafHealthError("causal context availability timestamps must be finite and unique")
    values = work.loc[:, list(selected)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise CausalLeafHealthError("causal context contains a non-finite declared value")
    timestamps = work["regime_available_utc"].map(lambda value: int(pd.Timestamp(value).value)).to_numpy(dtype=np.int64)
    order = np.argsort(timestamps, kind="stable")
    return timestamps[order], values[order], selected


def _asof_context(times: np.ndarray, values: np.ndarray, timestamp_ns: int) -> tuple[int, np.ndarray]:
    position = int(np.searchsorted(times, int(timestamp_ns), side="right") - 1)
    if position < 0:
        raise CausalLeafHealthError("candidate has no prior-available causal context row")
    return int(times[position]), values[position]


def _direct_field_names(section: str, metrics: Sequence[str]) -> tuple[str, ...]:
    names: list[str] = []
    for head in HEADS:
        for direction in DIRECTIONS:
            prefix = f"base_health__{section}__{head}__{direction}"
            names.extend(f"{prefix}__{metric}" for metric in metrics)
            names.append(f"{prefix}__active_abs_contribution")
    return tuple(names)


_DIRECT_NAMES = (*_direct_field_names("h1", _H1_METRICS), *_direct_field_names("h2", _H2_METRICS), *_direct_field_names("h3", _H3_METRICS))

# These are deliberately *sufficient statistics*, not inference fields.  A
# scope sees one semantic head at a time, whereas S2 needs one entropy over
# every token-free family contribution belonging to the full candidate
# identity.  Both statistics are additive across heads/scopes, so the final
# candidate join can form the exact entropy without retaining a leaf token or
# a per-family candidate table.
_ENTROPY_MASS = "__reasoning_entropy_abs_mass"
_ENTROPY_MASS_LOG_MASS = "__reasoning_entropy_abs_mass_log_mass"
REASONING_ENTROPY_FIELD = "base_reasoning__family_contribution_entropy"


def _entropy_statistics(contributions: np.ndarray) -> tuple[np.float32, np.float32]:
    """Return additive Shannon-entropy statistics for one candidate/head.

    ``contributions`` are current-tree contributions already present in the
    sealed feature-event stream.  The eventual entropy uses
    ``p_i = abs(c_i) / sum(abs(c))`` over every family/head for the full strict
    candidate identity.  Zero mass is represented by two zero statistics and
    is later mapped to entropy zero.  This is vectorised over the candidate's
    family vector and deliberately has no label/outcome dependency.
    """

    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise CausalLeafHealthError("family contribution entropy requires a finite one-dimensional vector")
    mass = np.abs(values)
    total = float(mass.sum(dtype=np.float64))
    if total <= 0.0:
        return np.float32(0.0), np.float32(0.0)
    positive = mass > 0.0
    moment = float(np.dot(mass[positive], np.log(mass[positive])))
    return np.float32(total), np.float32(moment)


def _entropy_statistics_block(
    contributions: np.ndarray, offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised sufficient statistics for all candidate groups in a block."""

    values = np.asarray(contributions, dtype=np.float64)
    boundaries = np.asarray(offsets, dtype=np.int64)
    if boundaries.ndim != 1 or len(boundaries) < 1 or boundaries[0] != 0 or boundaries[-1] != len(values):
        raise CausalLeafHealthError("entropy block offsets do not cover the contribution vector")
    if not np.isfinite(values).all() or np.any(np.diff(boundaries) < 0):
        raise CausalLeafHealthError("entropy block requires finite contributions and monotonic offsets")
    count = len(boundaries) - 1
    if count == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    widths = np.diff(boundaries)
    mass = np.abs(values)
    log_mass = np.zeros_like(mass)
    positive = mass > 0.0
    log_mass[positive] = mass[positive] * np.log(mass[positive])
    # Event groups normally contain at least one family.  The fallback keeps
    # the zero-family edge case explicit without introducing a large
    # per-contribution candidate-index allocation.
    if np.all(widths > 0):
        starts = boundaries[:-1]
        return (
            np.add.reduceat(mass, starts).astype(np.float32, copy=False),
            np.add.reduceat(log_mass, starts).astype(np.float32, copy=False),
        )
    totals = np.zeros(count, dtype=np.float32)
    moments = np.zeros(count, dtype=np.float32)
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        if end > start:
            totals[index] = mass[start:end].sum(dtype=np.float64)
            moments[index] = log_mass[start:end].sum(dtype=np.float64)
    return totals, moments


class _ParquetRows:
    """Small bounded row writer which preserves a stable Parquet schema."""

    def __init__(self, path: Path, *, flush_rows: int = _DIRECT_WRITER_ROWS) -> None:
        self.path = path
        self.flush_rows = int(flush_rows)
        self.rows: list[dict[str, Any]] = []
        self.writer: pq.ParquetWriter | None = None
        self.count = 0

    def add(self, row: Mapping[str, Any]) -> None:
        self.rows.append(dict(row))
        if len(self.rows) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pandas(pd.DataFrame(self.rows), preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table)
        self.count += len(self.rows)
        self.rows.clear()

    def close(self, *, empty_columns: Sequence[str] = ()) -> int:
        self.flush()
        if self.writer is None:
            pd.DataFrame({name: pd.Series(dtype="float32") for name in empty_columns}).to_parquet(self.path, index=False, compression="zstd")
        else:
            self.writer.close()
        return self.count


@dataclass
class _PeriodAccumulator:
    rows: int = 0
    successes: float = 0.0
    prediction_sum: float = 0.0
    net_sum: float = 0.0
    expected_sum: float = 0.0
    residual_sum: float = 0.0
    residual_sq_sum: float = 0.0
    calibration_sum: float = 0.0
    false_positive_loss_sum: float = 0.0
    timestamp_count: int = 0
    day_count: int = 0
    last_timestamp: int = np.iinfo(np.int64).min
    last_day: int = np.iinfo(np.int32).min
    assets: set[str] = field(default_factory=set)
    max_label_timestamp: int = np.iinfo(np.int64).min

    def update(self, group: ContributionEventGroup) -> None:
        success = float(group.semantic_label)
        prediction = float(group.head_prediction)
        net = float(group.net_bps)
        expected = float(group.base_expected_bps)
        residual = net - expected
        self.rows += 1
        self.successes += success
        self.prediction_sum += prediction
        self.net_sum += net
        self.expected_sum += expected
        self.residual_sum += residual
        self.residual_sq_sum += residual * residual
        self.calibration_sum += success - prediction
        if prediction >= 0.5 and success <= 0.0 and net < 0.0:
            self.false_positive_loss_sum += -net
        if self.last_timestamp != int(group.decision_timestamp_ns):
            self.timestamp_count += 1
            self.last_timestamp = int(group.decision_timestamp_ns)
        day = int(group.decision_timestamp_ns // (24 * 3600 * 1_000_000_000))
        if self.last_day != day:
            self.day_count += 1
            self.last_day = day
        self.assets.add(str(group.asset))
        self.max_label_timestamp = max(self.max_label_timestamp, int(group.timestamp_ns))


@dataclass
class _ScopeState:
    lookup: MutableMapping[tuple[str, str], int]
    reverse: list[tuple[str, str] | None]
    asset_lookup: dict[str, int]
    arrays: dict[str, np.ndarray]
    side_rows: float = 0.0
    side_successes: float = 0.0
    h3_records: dict[int, deque[tuple[np.ndarray, float]]] = field(default_factory=dict)
    # H3 is deliberately *month-frozen*.  A label that resolves during the
    # score month must update H1 immediately, but cannot enter that month's
    # H3 ridge.  Retain it in a small bounded pending queue and promote it
    # only at the following score-month boundary.  This is separate from the
    # H1 state because H1 is event-prequential whereas H3 is calendar-frozen.
    h3_pending: dict[int, deque[tuple[np.ndarray, float]]] = field(default_factory=dict)


def _resize_state(state: _ScopeState, *, family_capacity: int | None = None, asset_capacity: int | None = None) -> None:
    old_capacity = len(state.arrays["family_rows"])
    old_assets = state.arrays["family_asset_seen"].shape[1]
    new_capacity = max(old_capacity, int(family_capacity or old_capacity))
    new_assets = max(old_assets, int(asset_capacity or old_assets))
    if new_capacity == old_capacity and new_assets == old_assets:
        return
    replacement = allocate_h1_state(new_capacity, new_assets)
    for name, value in state.arrays.items():
        if name == "family_asset_seen":
            replacement[name][:old_capacity, :old_assets] = value
        else:
            replacement[name][:old_capacity] = value
    state.arrays = replacement


def _sync_codes(state: _ScopeState) -> None:
    required = len(state.lookup)
    if required > len(state.arrays["family_rows"]):
        _resize_state(state, family_capacity=max(required, len(state.arrays["family_rows"]) * 2))
    if required > len(state.reverse):
        state.reverse = [None] * required
        for key, code in state.lookup.items():
            state.reverse[int(code)] = key


def _asset_codes(state: _ScopeState, groups: Sequence[ContributionEventGroup]) -> np.ndarray:
    for group in groups:
        if group.asset not in state.asset_lookup:
            state.asset_lookup[group.asset] = len(state.asset_lookup)
    required = len(state.asset_lookup)
    if required > state.arrays["family_asset_seen"].shape[1]:
        _resize_state(state, asset_capacity=max(required, state.arrays["family_asset_seen"].shape[1] * 2))
    return np.asarray([state.asset_lookup[group.asset] for group in groups], dtype=np.int32)


def _pack(groups: Sequence[ContributionEventGroup]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.zeros(len(groups) + 1, dtype=np.int64)
    for index, group in enumerate(groups):
        offsets[index + 1] = offsets[index] + len(group.family_codes)
    if int(offsets[-1]) == 0:
        raise CausalLeafHealthError("contribution event candidate has no family rows")
    return (
        np.concatenate([group.family_codes for group in groups]).astype(np.int32, copy=False),
        np.concatenate([group.directions for group in groups]).astype(np.int8, copy=False),
        np.concatenate([group.contributions for group in groups]).astype(np.float32, copy=False),
        offsets,
    )


def _scope_months(streams: StrictContributionEventStreams, *, dataset: str, key: tuple[str, str, str]) -> list[str]:
    contract, side, head = key
    part = streams.part_index
    selected = part.loc[
        part["dataset"].eq(dataset)
        & part["contract"].astype(str).eq(contract)
        & part["side"].astype(str).eq(side)
        & part["head"].astype(str).eq(head),
        "month",
    ].astype(str).unique().tolist()
    return sorted(selected)


def _iter_scope_blocks(
    streams: StrictContributionEventStreams, *, dataset: str, key: tuple[str, str, str],
    family_lookup: MutableMapping[tuple[str, str], int], batch_rows: int, max_open_parts: int,
) -> Iterator[ContributionEventTimestampBlock]:
    """Read one decision month at a time and prove cross-month ordering."""

    previous: int | None = None
    for month in _scope_months(streams, dataset=dataset, key=key):
        for block in iter_contribution_event_timestamp_blocks(
            streams, dataset=dataset, contract=key[0], side=key[1], head=key[2], months=[month],
            family_lookup=family_lookup, batch_rows=int(batch_rows), max_open_parts=int(max_open_parts),
        ):
            if previous is not None and block.timestamp_ns < previous:
                raise CausalLeafHealthError(
                    "sidecar monthly pieces are not globally ordered; a wider bounded merge is required before causal health"
                )
            previous = int(block.timestamp_ns)
            yield block


def _prior_series(store: StrictEventStore, temporary: Path, *, memory_limit: str, temp_disk_limit: str) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], int]:
    index = _scope_rows(store, "resolution_order")
    paths = _paths_literal(store.root, index["path"].astype(str))
    output = temporary / "global_h1_resolution.parquet"
    connection = duckdb.connect(database=str(temporary / "global_priors.duckdb"))
    try:
        connection.execute("PRAGMA threads=2")
        connection.execute(f"PRAGMA memory_limit={_literal(memory_limit)}")
        connection.execute(f"PRAGMA temp_directory={_literal(str(temporary / 'global_prior_tmp'))}")
        connection.execute(f"SET max_temp_directory_size={_literal(temp_disk_limit)}")
        rows = _candidate_global_resolution(connection, paths, output)
    finally:
        connection.close()
        shutil.rmtree(temporary / "global_prior_tmp", ignore_errors=True)
        (temporary / "global_priors.duckdb").unlink(missing_ok=True)
    frame = pd.read_parquet(output)
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for contract, group in frame.groupby("feature_contract_sha256", observed=True, sort=False):
        ordered = group.sort_values("label_available_ts", kind="stable")
        result[str(contract)] = (
            pd.to_datetime(ordered["label_available_ts"], utc=True).map(lambda value: int(pd.Timestamp(value).value)).to_numpy(dtype=np.int64),
            pd.to_numeric(ordered["rows"], errors="raise").to_numpy(dtype=np.float64),
            pd.to_numeric(ordered["successes"], errors="raise").to_numpy(dtype=np.float64),
        )
    return result, int(rows)


def _prior_before(series: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]], contract: str, timestamp_ns: int) -> tuple[float, float]:
    values = series.get(contract)
    if values is None:
        return 0.0, 0.0
    times, rows, successes = values
    position = int(np.searchsorted(times, int(timestamp_ns), side="left") - 1)
    if position < 0:
        return 0.0, 0.0
    return float(rows[position]), float(successes[position])


def _family_h1_values(state: _ScopeState, code: int, *, global_rows: float, global_successes: float, config: CausalLeafHealthConfig) -> np.ndarray:
    arrays = state.arrays
    rows = float(arrays["family_rows"][code])
    successes = float(arrays["family_successes"][code])
    global_mean = (global_successes + float(config.global_alpha)) / max(global_rows + float(config.global_alpha) + float(config.global_beta), 1e-12)
    side_mean = (state.side_successes + float(config.side_head_prior_strength) * global_mean) / max(state.side_rows + float(config.side_head_prior_strength), 1e-12)
    alpha = successes + float(config.family_prior_strength) * side_mean
    beta = rows - successes + float(config.family_prior_strength) * (1.0 - side_mean)
    total = max(alpha + beta, 1e-12)
    posterior = alpha / total
    lower = max(0.0, posterior - 1.96 * math.sqrt(max(posterior * (1.0 - posterior) / (total + 1.0), 0.0)))
    divisor = max(rows, 1.0)
    support = min(1.0, min(
        float(arrays["family_timestamps"][code]) / float(config.min_timestamp_support),
        float(arrays["family_days"][code]) / float(config.min_day_support),
        float(arrays["family_symbols"][code]) / float(config.min_symbol_support),
    ))
    return np.asarray([
        posterior, lower, rows, float(arrays["family_timestamps"][code]), float(arrays["family_days"][code]),
        float(arrays["family_symbols"][code]), support,
        successes / divisor - float(arrays["family_predictions"][code]) / divisor,
        float(arrays["family_nets"][code]) / divisor - float(arrays["family_expecteds"][code]) / divisor,
        float(arrays["family_false_positive_losses"][code]) / divisor,
    ], dtype=np.float64)


def _empty_h2() -> np.ndarray:
    return np.zeros(len(_H2_METRICS), dtype=np.float64)


def _period_summaries(
    streams: StrictContributionEventStreams, *, key: tuple[str, str, str], state: _ScopeState,
    output_dir: Path, batch_rows: int, max_open_parts: int,
) -> tuple[list[Path], int]:
    """Aggregate one decision month at a time into compact, exact H2 inputs.

    The strict sidecar is source-month partitioned, so this pass never needs a
    scope-wide group-by.  We deliberately validate monotone decision times;
    that makes the compact last-seen timestamp/day counters exact rather than
    silently treating a reordered label stream as distinct support.
    """

    paths: list[Path] = []
    rows_written = 0
    for month in _scope_months(streams, dataset="resolution_event_order", key=key):
        values: dict[int, _PeriodAccumulator] = {}
        prior_decision: int | None = None
        period_end = _month_end_ns(month)
        for block in iter_contribution_event_timestamp_blocks(
            streams, dataset="resolution_event_order", contract=key[0], side=key[1], head=key[2],
            months=[month], family_lookup=state.lookup, batch_rows=int(batch_rows), max_open_parts=int(max_open_parts),
        ):
            _sync_codes(state)
            for group in block.groups:
                if prior_decision is not None and group.decision_timestamp_ns < prior_decision:
                    raise CausalLeafHealthError(
                        "label resolution order regresses in decision time; exact H1/H2 support cannot use last-seen counters"
                    )
                prior_decision = int(group.decision_timestamp_ns)
                # H2 is a completed-period statistic.  A path which only
                # resolves after the decision period has closed is not part
                # of that period's frozen support/economic summary (even if
                # its entry was near the prior month end).  H1 still learns
                # it prequentially; this filter is H2-only.
                if int(group.label_available_timestamp_ns) >= period_end:
                    continue
                # The sealed sidecar builder proves a candidate/family is
                # already collapsed.  Do not re-sort or hash every candidate
                # here: that would turn this bounded linear pass back into the
                # dominant hot path.
                for code in group.family_codes:
                    accumulator = values.setdefault(int(code), _PeriodAccumulator())
                    accumulator.update(group)
        if not values:
            continue
        records: list[dict[str, Any]] = []
        for code, value in values.items():
            records.append({
                "family_code": np.int32(code), "period": month, "period_end_ns": np.int64(period_end),
                "rows": np.int32(value.rows), "successes": np.float64(value.successes),
                "prediction_sum": np.float64(value.prediction_sum), "net_sum": np.float64(value.net_sum),
                "expected_sum": np.float64(value.expected_sum), "residual_sum": np.float64(value.residual_sum),
                "residual_sq_sum": np.float64(value.residual_sq_sum), "calibration_sum": np.float64(value.calibration_sum),
                "false_positive_loss_sum": np.float64(value.false_positive_loss_sum),
                "independent_timestamps": np.int32(value.timestamp_count), "trading_days": np.int32(value.day_count),
                "symbols": np.int32(len(value.assets)), "max_label_available_ns": np.int64(value.max_label_timestamp),
            })
        path = output_dir / f"period_{month}.parquet"
        pd.DataFrame(records).to_parquet(path, index=False, compression="zstd")
        paths.append(path)
        rows_written += len(records)
    return paths, rows_written


def _h2_snapshot(
    summaries: pd.DataFrame, *, score_month_start_ns: int, state: _ScopeState,
    config: CausalLeafHealthConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build the exact month-frozen H2 surface from previously closed periods.

    The close condition is evaluated against the calendar score-month start,
    not the timestamp of the first candidate encountered in that month.  The
    latter would leak a period into later-in-the-month candidates whenever the
    first candidate arrived after the configured close lag.
    """

    capacity = len(state.arrays["family_rows"])
    values = np.zeros((capacity, len(_H2_METRICS)), dtype=np.float64)
    # The reference's no-closed-period state is deliberately an explicit
    # support failure, not a neutral zero.  All other unavailable H2 metrics
    # coalesce to zero only at contribution-weighted feature aggregation.
    values[:, _H2_METRICS.index("support_failure")] = 1.0
    if summaries.empty:
        return values, []
    close_before = int(score_month_start_ns) - int(config.period_close_lag_hours) * 3600 * 1_000_000_000
    eligible = summaries.loc[
        summaries["period_end_ns"].astype("int64").le(close_before)
        & summaries["max_label_available_ns"].astype("int64").lt(int(score_month_start_ns))
    ]
    if eligible.empty:
        return values, []
    raw: dict[int, dict[str, float | str]] = {}
    by_bucket: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for raw_code, group in eligible.groupby("family_code", sort=False, observed=True):
        code = int(raw_code)
        period = group.sort_values("period", kind="stable")
        period_count = float(len(period))
        supported = period.loc[
            period["rows"].astype(int).ge(int(config.min_period_rows))
            & period["independent_timestamps"].astype(int).ge(int(config.min_timestamp_support))
            & period["trading_days"].astype(int).ge(int(config.min_day_support))
            & period["symbols"].astype(int).ge(int(config.min_symbol_support))
        ]
        support_failure = 1.0 - float(len(supported)) / max(period_count, 1.0)
        if supported.empty:
            item: dict[str, float | str] = {
                "period_count": period_count, "supported_period_count": 0.0,
                "observed_variance": np.nan, "sampling_variance": np.nan, "excess_variance": np.nan,
                "sign_reversal_rate": np.nan, "worst_damage_bps": np.nan, "calibration_drift": np.nan,
                "support_failure": support_failure, "instability": np.nan, "classification": "LOW_SUPPORT_UNCERTAIN",
            }
        else:
            rows = supported["rows"].to_numpy(dtype=np.float64)
            effects = supported["residual_sum"].to_numpy(dtype=np.float64) / np.maximum(rows, 1.0)
            residual_sq = supported["residual_sq_sum"].to_numpy(dtype=np.float64)
            variances = np.maximum(residual_sq / np.maximum(rows, 1.0) - effects * effects, 0.0)
            errors = np.where(rows >= 2.0, np.sqrt(variances / np.maximum(rows, 1.0)), np.nan)
            centre = float(np.average(effects, weights=rows))
            observed = float(np.average((effects - centre) ** 2, weights=rows))
            sampling = float(np.nanmean(errors ** 2)) if np.isfinite(errors).any() else 0.0
            excess = max(0.0, observed - sampling)
            credible = np.abs(effects) > 1.96 * np.nan_to_num(errors, nan=np.inf)
            reversal: list[float] = []
            for left, right, left_ok, right_ok in zip(effects[:-1], effects[1:], credible[:-1], credible[1:], strict=True):
                if bool(left_ok) and bool(right_ok):
                    reversal.append(float(np.sign(left) != np.sign(right)))
            sign_reversal = float(np.mean(reversal)) if reversal else 0.0
            worst_damage = max(0.0, -float(np.nanmin(effects)))
            calibration = supported["calibration_sum"].to_numpy(dtype=np.float64) / np.maximum(rows, 1.0)
            calibration_drift = float(np.average(np.abs(calibration), weights=rows))
            classification = "STABLE_PORTABLE"
            if len(supported) < int(config.min_periods):
                classification = "LOW_SUPPORT_UNCERTAIN"
            elif support_failure >= float(config.h2_support_failure_threshold):
                classification = "SUPPORT_SHIFT_ONLY"
            elif calibration_drift >= float(config.h2_calibration_drift_threshold):
                classification = "CALIBRATION_DRIFT"
            elif sign_reversal > 0.0 or excess > 0.0:
                classification = "GLOBAL_PERIOD_SENSITIVE"
            item = {
                "period_count": period_count, "supported_period_count": float(len(supported)),
                "observed_variance": observed, "sampling_variance": sampling, "excess_variance": excess,
                "sign_reversal_rate": sign_reversal, "worst_damage_bps": worst_damage,
                "calibration_drift": calibration_drift, "support_failure": support_failure,
                "instability": np.nan, "classification": classification,
            }
            bucket = int(np.floor(np.log2(max(int(state.arrays["family_rows"][code]), 1))))
            by_bucket[bucket].append((code, excess))
        raw[code] = item
    robust_z: dict[int, float] = {}
    for entries in by_bucket.values():
        source = np.asarray([value for _, value in entries], dtype=np.float64)
        median = float(np.median(source))
        scale = max(float(np.median(np.abs(source - median))) * 1.4826, 1e-8)
        for code, value in entries:
            robust_z[code] = float((value - median) / scale)
    damage_scale = max(float(np.nanmedian([
        float(item["worst_damage_bps"]) for item in raw.values() if np.isfinite(float(item["worst_damage_bps"]))
    ] or [1.0])), 1.0)
    audit: list[dict[str, Any]] = []
    for code, item in raw.items():
        z = robust_z.get(code, np.nan)
        instability = (
            float(np.nan_to_num(z, nan=0.0))
            + float(np.nan_to_num(item["sign_reversal_rate"], nan=0.0))
            + float(np.nan_to_num(item["worst_damage_bps"], nan=0.0)) / damage_scale
            + 0.5 * float(item["support_failure"])
        )
        item["robust_z_excess_variance"] = z
        item["instability"] = instability
        if item["classification"] != "LOW_SUPPORT_UNCERTAIN" and instability > float(config.h2_instability_threshold):
            item["classification"] = "GLOBAL_PERIOD_SENSITIVE"
        ordered = [float(np.nan_to_num(item[name], nan=0.0)) for name in _H2_METRICS]
        values[code, :] = np.asarray(ordered, dtype=np.float64)
        audit.append({"family_code": np.int32(code), **item})
    return values, audit


def _h3_models(state: _ScopeState, config: CausalLeafHealthConfig) -> dict[int, Any]:
    result: dict[int, Any] = {}
    for code, records in state.h3_records.items():
        model = _fit_ridge(records, config)
        if model is not None:
            result[code] = model
    return result


def _commit_h3_pending(state: _ScopeState, config: CausalLeafHealthConfig) -> None:
    """Make prior score-month labels available to the next H3 snapshot.

    ``h3_pending`` contains only labels that resolved after the previous
    month boundary.  Committing them atomically before the next H3 fit is the
    bounded-memory equivalent of the reference condition
    ``label_available_ts < score_month_start``.  The max-row cap is applied
    after the merge, retaining the most recently resolved admissible rows as
    the vectorised reference does.
    """

    maximum = int(config.h3_max_rows_per_family)
    for code, pending in state.h3_pending.items():
        if not pending:
            continue
        records = state.h3_records.setdefault(code, deque())
        records.extend(pending)
        while len(records) > maximum:
            records.popleft()
    state.h3_pending.clear()


def _h3_values(
    state: _ScopeState, models: Mapping[int, Any], *, context: np.ndarray,
    active: bool, selected_context_codes: set[int], config: CausalLeafHealthConfig,
) -> np.ndarray:
    values = np.zeros((len(state.arrays["family_rows"]), len(_H3_METRICS)), dtype=np.float64)
    if not active:
        return values
    for code in selected_context_codes:
        model = models.get(code)
        if model is None:
            continue
        error = float(model.predict(context))
        compatibility = float(np.exp(-abs(error) / max(float(model.residual_scale), 1e-6)))
        confidence = float(model.rows / (model.rows + float(config.h3_min_rows)))
        values[code, :] = (1.0, compatibility, error, confidence, (1.0 - compatibility) * confidence)
    return values


def _selection_codes(
    state: _ScopeState, *, key: tuple[str, str, str],
    selection: frozenset[tuple[str, str, str, str, str]],
) -> set[int]:
    result: set[int] = set()
    for code, value in enumerate(state.reverse):
        if value is not None and (*key, value[0], value[1]) in selection:
            result.add(code)
    return result


def _update_resolution_block(
    block: ContributionEventTimestampBlock, *, state: _ScopeState,
    context_times: np.ndarray, context_values: np.ndarray, selected_context_codes: set[int],
    config: CausalLeafHealthConfig, last_decision_timestamp: int | None,
    h3_commit: bool,
) -> int | None:
    """Apply a complete resolved timestamp after feature events at that time."""

    groups = block.groups
    _sync_codes(state)
    decision_times = np.asarray([item.decision_timestamp_ns for item in groups], dtype=np.int64)
    if np.any(decision_times[1:] < decision_times[:-1]) or (last_decision_timestamp is not None and int(decision_times[0]) < last_decision_timestamp):
        raise CausalLeafHealthError(
            "resolution stream does not preserve decision-time order; exact timestamp/day support requires a bounded regrouping repair"
        )
    codes, _, contributions, offsets = _pack(groups)
    day = (decision_times // (24 * 3600 * 1_000_000_000)).astype(np.int32)
    asset_codes = _asset_codes(state, groups)
    arrays = state.arrays
    update_h1_block(
        codes, contributions, offsets,
        np.asarray([item.semantic_label for item in groups], dtype=np.float64),
        np.asarray([item.head_prediction for item in groups], dtype=np.float64),
        np.asarray([item.net_bps for item in groups], dtype=np.float64),
        np.asarray([item.base_expected_bps for item in groups], dtype=np.float64),
        decision_times, day, asset_codes,
        arrays["family_rows"], arrays["family_successes"], arrays["family_predictions"],
        arrays["family_nets"], arrays["family_expecteds"], arrays["family_false_positive_losses"],
        arrays["family_timestamps"], arrays["family_days"], arrays["family_symbols"],
        arrays["family_last_timestamp"], arrays["family_last_day"], arrays["family_asset_seen"],
    )
    state.side_rows += float(len(groups))
    state.side_successes += float(sum(item.semantic_label for item in groups))
    if selected_context_codes:
        for item in groups:
            _, context = _asof_context(context_times, context_values, item.feature_generation_timestamp_ns)
            target = float(item.net_bps) - float(item.base_expected_bps)
            for code in item.family_codes:
                code_i = int(code)
                if code_i in selected_context_codes:
                    records = (
                        state.h3_records.setdefault(code_i, deque())
                        if h3_commit
                        else state.h3_pending.setdefault(code_i, deque())
                    )
                    records.append((context.astype(np.float64, copy=True), target))
                    while len(records) > int(config.h3_max_rows_per_family):
                        records.popleft()
    return int(decision_times[-1])


def _direct_row(
    group: ContributionEventGroup, *, side: str, head: str,
    h1: np.ndarray, auxiliary: np.ndarray, entropy_mass: np.float32 | None = None,
    entropy_moment: np.float32 | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_id": group.candidate_id, "decision_ts": _timestamp(group.decision_timestamp_ns),
        "side_name": side, "fold_id": group.fold_id, "transport": group.transport,
        "meta_partition": group.meta_partition,
    }
    # The kernel direction convention is 0=negative, 1=positive; the stable
    # health feature contract uses positive then negative.
    for direction, direction_code in (("positive", 1), ("negative", 0)):
        h1_prefix = f"base_health__h1__{head}__{direction}"
        h2_prefix = f"base_health__h2__{head}__{direction}"
        h3_prefix = f"base_health__h3__{head}__{direction}"
        for metric_index, metric in enumerate(_H1_METRICS):
            row[f"{h1_prefix}__{metric}"] = np.float32(h1[direction_code, metric_index])
        row[f"{h1_prefix}__active_abs_contribution"] = np.float32(h1[direction_code, H1_METRIC_COUNT])
        for metric_index, metric in enumerate(_H2_METRICS):
            row[f"{h2_prefix}__{metric}"] = np.float32(auxiliary[direction_code, metric_index])
        row[f"{h2_prefix}__active_abs_contribution"] = np.float32(auxiliary[direction_code, -1])
        base = len(_H2_METRICS)
        for metric_index, metric in enumerate(_H3_METRICS):
            row[f"{h3_prefix}__{metric}"] = np.float32(auxiliary[direction_code, base + metric_index])
        row[f"{h3_prefix}__active_abs_contribution"] = np.float32(auxiliary[direction_code, -1])
    if entropy_mass is None or entropy_moment is None:
        entropy_mass, entropy_moment = _entropy_statistics(group.contributions)
    row[_ENTROPY_MASS] = entropy_mass
    row[_ENTROPY_MASS_LOG_MASS] = entropy_moment
    return row


def _selected_state_row(
    group: ContributionEventGroup, *, family_index: int, code: int,
    key: tuple[str, str, str], state: _ScopeState, h1_values: np.ndarray,
    h2_values: np.ndarray, h3_values: np.ndarray, context_time: int, context: np.ndarray,
    context_columns: Sequence[str],
    h4_active: bool, h5_active: bool,
) -> dict[str, Any]:
    signature, direction = state.reverse[code] or ("", "")
    contract, side, head = key
    row: dict[str, Any] = {
        "candidate_id": group.candidate_id, "decision_ts": _timestamp(group.decision_timestamp_ns),
        "side_name": side, "fold_id": group.fold_id, "transport": group.transport,
        "meta_partition": group.meta_partition, "feature_generation_ts": _timestamp(group.feature_generation_timestamp_ns),
        "label_available_ts": _timestamp(group.label_available_timestamp_ns),
        "feature_contract_sha256": contract, "head_name": head, "rule_signature": signature,
        "contribution_direction": direction,
        "family_ensemble_tree_contribution": np.float32(group.contributions[family_index]),
        "regime_available_utc": _timestamp(context_time),
        "h4_selection_active": np.float32(float(h4_active)),
        "h5_selection_active": np.float32(float(h5_active)),
    }
    row.update({f"h1_{metric}": np.float32(h1_values[index]) for index, metric in enumerate(_H1_METRICS)})
    row.update({f"h2_{metric}": np.float32(h2_values[index]) for index, metric in enumerate(_H2_METRICS)})
    row.update({f"h3_{metric}": np.float32(h3_values[index]) for index, metric in enumerate(_H3_METRICS)})
    row.update({str(column): np.float32(value) for column, value in zip(context_columns, context, strict=True)})
    return row


def _process_scope(
    streams: StrictContributionEventStreams, *, key: tuple[str, str, str], output_dir: Path,
    global_prior: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]], context_times: np.ndarray,
    context_values: np.ndarray, context_columns: Sequence[str], config: CausalLeafHealthConfig,
    batch_rows: int, max_open_parts: int, max_selected_state_rows: int,
) -> dict[str, Any]:
    """Materialise one scope with at most a monthly sidecar window in memory."""

    contract, side, head = key
    state = _ScopeState({}, [], {}, allocate_h1_state(4_096, 8))
    period_dir = output_dir / "period_parts"
    period_dir.mkdir(exist_ok=True)
    period_paths, period_rows = _period_summaries(
        streams, key=key, state=state, output_dir=period_dir,
        batch_rows=batch_rows, max_open_parts=max_open_parts,
    )
    _sync_codes(state)
    if period_paths:
        summary = pd.concat([pd.read_parquet(path) for path in period_paths], ignore_index=True)
    else:
        summary = pd.DataFrame()
    selected_context = _selection_codes(state, key=key, selection=config.selected_context_families)
    selected_covariance = _selection_codes(state, key=key, selection=config.selected_covariance_families)
    selected_relationship = _selection_codes(state, key=key, selection=config.selected_relationship_families)
    selected_any = selected_context | selected_covariance | selected_relationship
    direct_path = output_dir / "direct.parquet"
    selected_path = output_dir / "selected_states.parquet"
    denominator_path = output_dir / "denominators.parquet"
    portability_path = output_dir / "portability.parquet"
    direct_writer = _ParquetRows(direct_path)
    selected_writer = _ParquetRows(selected_path)
    denominator_writer = _ParquetRows(denominator_path)
    portability_writer = _ParquetRows(portability_path)
    resolutions = _iter_scope_blocks(
        streams, dataset="resolution_event_order", key=key, family_lookup=state.lookup,
        batch_rows=batch_rows, max_open_parts=max_open_parts,
    )
    next_resolution = next(resolutions, None)
    last_resolution_decision: int | None = None
    snapshot_month: str | None = None
    h2_values = np.zeros((len(state.arrays["family_rows"]), len(_H2_METRICS)), dtype=np.float64)
    h3_values = np.zeros((len(state.arrays["family_rows"]), len(_H3_METRICS)), dtype=np.float64)
    h3_models: dict[int, Any] = {}
    selected_count = 0
    portability_count = 0
    for feature_block in _iter_scope_blocks(
        streams, dataset="feature_event_order", key=key, family_lookup=state.lookup,
        batch_rows=batch_rows, max_open_parts=max_open_parts,
    ):
        _sync_codes(state)
        month = _month_key(feature_block.timestamp_ns)
        if month != snapshot_month:
            # Finish the prior score month first.  Every label that resolved
            # during it can train the next month's H3 model, but none may
            # alter the already-frozen model that scored that month.
            _commit_h3_pending(state, config)
            month_start = _utc_ns(f"{month}-01T00:00:00Z")
            while next_resolution is not None and next_resolution.timestamp_ns < month_start:
                last_resolution_decision = _update_resolution_block(
                    next_resolution, state=state, context_times=context_times, context_values=context_values,
                    selected_context_codes=selected_context, config=config,
                    last_decision_timestamp=last_resolution_decision, h3_commit=True,
                )
                next_resolution = next(resolutions, None)
            h2_values, h2_audit = _h2_snapshot(
                summary, score_month_start_ns=month_start, state=state, config=config,
            )
            h3_models = _h3_models(state, config)
            _, month_context = _asof_context(context_times, context_values, feature_block.timestamp_ns)
            h3_values = _h3_values(
                state, h3_models, context=month_context,
                active=_selection_active(feature_block.timestamp_ns, config),
                selected_context_codes=selected_context, config=config,
            )
            for audit in h2_audit:
                code = int(audit.pop("family_code"))
                signature, direction = state.reverse[code] or ("", "")
                portability_writer.add({
                    "score_month": month, "feature_contract_sha256": contract, "side_name": side,
                    "head_name": head, "rule_signature": signature, "contribution_direction": direction,
                    **audit, "is_portable": bool(audit["classification"] == "STABLE_PORTABLE"),
                })
                portability_count += 1
            snapshot_month = month
        # H1 sees every previously-resolved label event immediately.  H3
        # receives these values only at the following calendar boundary.
        while next_resolution is not None and next_resolution.timestamp_ns < feature_block.timestamp_ns:
            last_resolution_decision = _update_resolution_block(
                next_resolution, state=state, context_times=context_times, context_values=context_values,
                selected_context_codes=selected_context, config=config,
                last_decision_timestamp=last_resolution_decision, h3_commit=False,
            )
            next_resolution = next(resolutions, None)
        # H3 is a frozen monthly model but the context is candidate-as-of,
        # not a single monthly state.  Recompute only the selected rows here.
        context_time, context = _asof_context(context_times, context_values, feature_block.timestamp_ns)
        active = _selection_active(feature_block.timestamp_ns, config)
        h3_values = _h3_values(state, h3_models, context=context, active=active, selected_context_codes=selected_context, config=config)
        groups = feature_block.groups
        codes, directions, contributions, offsets = _pack(groups)
        entropy_masses, entropy_moments = _entropy_statistics_block(contributions, offsets)
        global_rows, global_successes = _prior_before(global_prior, contract, feature_block.timestamp_ns)
        arrays = state.arrays
        h1 = score_h1_block(
            codes, directions, contributions, offsets,
            arrays["family_rows"], arrays["family_successes"], arrays["family_predictions"],
            arrays["family_nets"], arrays["family_expecteds"], arrays["family_false_positive_losses"],
            arrays["family_timestamps"], arrays["family_days"], arrays["family_symbols"],
            global_rows, global_successes, state.side_rows, state.side_successes,
            float(config.global_alpha), float(config.global_beta), float(config.side_head_prior_strength),
            float(config.family_prior_strength), float(config.min_timestamp_support),
            float(config.min_day_support), float(config.min_symbol_support),
        )
        auxiliary = score_auxiliary_block(codes, directions, contributions, offsets, h2_values, h3_values)
        for index, group in enumerate(groups):
            direct_writer.add(_direct_row(
                group, side=side, head=head, h1=h1[index], auxiliary=auxiliary[index],
                entropy_mass=entropy_masses[index], entropy_moment=entropy_moments[index],
            ))
            if not selected_any:
                continue
            has_selected = any(int(code) in selected_any for code in group.family_codes)
            if has_selected:
                for direction, direction_code in (("positive", 1), ("negative", 0)):
                    denominator = float(h1[index, direction_code, H1_METRIC_COUNT])
                    if denominator > 0.0:
                        denominator_writer.add({
                            "candidate_id": group.candidate_id, "decision_ts": _timestamp(group.decision_timestamp_ns),
                            "side_name": side, "fold_id": group.fold_id, "transport": group.transport,
                            "meta_partition": group.meta_partition, "head_name": head,
                            "contribution_direction": direction, "full_abs_contribution": np.float32(denominator),
                        })
            for family_index, raw_code in enumerate(group.family_codes):
                code = int(raw_code)
                if code not in selected_any:
                    continue
                selected_count += 1
                if selected_count > int(max_selected_state_rows):
                    raise CausalLeafHealthError("selected H3/H4/H5 state audit exceeds max_selected_state_rows")
                selected_writer.add(_selected_state_row(
                    group, family_index=family_index, code=code, key=key, state=state,
                    h1_values=_family_h1_values(state, code, global_rows=global_rows, global_successes=global_successes, config=config),
                    h2_values=h2_values[code], h3_values=h3_values[code], context_time=context_time,
                    context=context, context_columns=context_columns, h4_active=active and code in selected_covariance,
                    h5_active=active and code in selected_relationship,
                ))
    # Process tail labels only to make the period source audit complete.  They
    # cannot affect a scored feature row already written.
    while next_resolution is not None:
        last_resolution_decision = _update_resolution_block(
            next_resolution, state=state, context_times=context_times, context_values=context_values,
            selected_context_codes=selected_context, config=config,
            last_decision_timestamp=last_resolution_decision, h3_commit=False,
        )
        next_resolution = next(resolutions, None)
    return {
        "key": key, "direct_path": direct_path, "selected_path": selected_path,
        "denominator_path": denominator_path, "portability_path": portability_path,
        "period_paths": period_paths, "period_rows": period_rows,
        "direct_rows": direct_writer.close(empty_columns=[*_IDENTITY]),
        "selected_rows": selected_writer.close(empty_columns=[*_IDENTITY]),
        "denominator_rows": denominator_writer.close(empty_columns=[*_IDENTITY]),
        "portability_rows": portability_writer.close(empty_columns=[]),
        "family_count": len(state.lookup), "family_reverse": tuple(state.reverse),
    }


def _merge_selected(
    paths: Sequence[Path], *, output: Path, context_columns: Sequence[str],
    memory_limit: str = "1GB", temp_disk_limit: str = "8GB",
) -> int:
    """Concatenate selected-family states without decoding them into pandas.

    The selection cap applies globally, but even a valid multi-million-row
    audit can be several GiB once the ten causal context fields are present.
    A pandas concat here would turn a deliberately bounded event pass into an
    unbounded final merge.  DuckDB copies physical row groups directly into a
    single provenance artifact under explicit memory/spill limits.
    """

    populated = [path for path in paths if path.is_file() and pq.ParquetFile(path).metadata.num_rows > 0]
    if not populated:
        fields = [
            *_IDENTITY, "feature_generation_ts", "label_available_ts", "feature_contract_sha256",
            "head_name", "rule_signature", "contribution_direction", "family_ensemble_tree_contribution",
            *[f"h1_{name}" for name in _H1_METRICS], *[f"h2_{name}" for name in _H2_METRICS],
            *[f"h3_{name}" for name in _H3_METRICS], "regime_available_utc", *context_columns,
            "h4_selection_active", "h5_selection_active",
        ]
        pd.DataFrame({field: pd.Series(dtype="float32") for field in fields}).to_parquet(output, index=False, compression="zstd")
        return 0
    database = output.parent / "selected_state_merge.duckdb"
    spill = output.parent / "selected_state_merge_tmp"
    paths_literal = "[" + ", ".join(_literal(str(path)) for path in populated) + "]"
    connection = duckdb.connect(database=str(database))
    try:
        connection.execute("PRAGMA threads=2")
        connection.execute(f"PRAGMA memory_limit={_literal(memory_limit)}")
        connection.execute(f"PRAGMA temp_directory={_literal(str(spill))}")
        connection.execute(f"SET max_temp_directory_size={_literal(temp_disk_limit)}")
        connection.execute(
            f"COPY (SELECT * FROM read_parquet({paths_literal}, union_by_name=true)) "
            f"TO {_literal(str(output))} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        return int(connection.execute(f"SELECT count(*) FROM read_parquet({_literal(str(output))})").fetchone()[0])
    finally:
        connection.close()
        shutil.rmtree(spill, ignore_errors=True)
        database.unlink(missing_ok=True)


def _identifier(value: str) -> str:
    """Quote a DuckDB identifier without treating it as SQL syntax."""

    return '"' + str(value).replace('"', '""') + '"'


def _size_limit_bytes(value: str) -> int:
    """Parse the small DuckDB-size surface used by bounded postprocessing."""

    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB|KIB|MIB|GIB|TIB)\s*", str(value), re.I)
    if match is None:
        raise CausalLeafHealthError(f"unsupported bounded temporary-disk limit: {value!r}")
    number = float(match.group(1))
    units = match.group(2).upper()
    multiplier = {
        "B": 1, "KB": 1_000, "MB": 1_000_000, "GB": 1_000_000_000, "TB": 1_000_000_000_000,
        "KIB": 1 << 10, "MIB": 1 << 20, "GIB": 1 << 30, "TIB": 1 << 40,
    }[units]
    result = int(number * multiplier)
    if result <= 0:
        raise CausalLeafHealthError("temporary-disk limit must be positive")
    return result


def _check_bounded_file(path: Path, *, maximum_bytes: int, label: str) -> None:
    if path.is_file() and path.stat().st_size > int(maximum_bytes):
        raise CausalLeafHealthError(
            f"{label} exceeds the declared bounded temporary-disk limit ({maximum_bytes} bytes)"
        )


def _empty_covariance(path: Path) -> None:
    pd.DataFrame(columns=[
        *_IDENTITY, "head_name", "contribution_direction", "rule_signature",
        "family_ensemble_tree_contribution",
    ]).to_parquet(path, index=False, compression="zstd")


def _empty_relationships(path: Path) -> None:
    pd.DataFrame(columns=[
        *_IDENTITY, "head_name", "contribution_direction", "rule_signature",
        "family_ensemble_tree_contribution", "relationship_pair", "relationship_break",
        "material_break", "portable_economic_weight",
    ]).to_parquet(path, index=False, compression="zstd")


class _StreamingCovarianceState:
    """The public covariance state builder, split into bounded row transitions.

    The numerical operations intentionally call the same private primitives as
    :func:`build_causal_leaf_covariance_state`.  This avoids a second, almost
    identical covariance implementation whose shrinkage or same-timestamp
    semantics could drift from the tested reference.
    """

    def __init__(self, fields: Sequence[str], config: CausalLeafCovarianceConfig) -> None:
        self.fields = tuple(map(str, fields))
        self.config = config
        self.states: dict[int, dict[str, dict[object, _EWMCovariance]]] = {
            horizon: {level: {} for level in ("family", "side_head", "global")}
            for horizon in HORIZONS
        }
        self.references: dict[int, dict[str, dict[object, _EWMCovariance]]] | None = None
        self.active_block: str | None = None
        self.seen_blocks: set[str] = set()
        self.timestamp_ns: int | None = None

    @staticmethod
    def _keys(family: str, side: str, head: str) -> dict[str, object]:
        return {
            "family": family,
            "side_head": (side, head),
            "global": "__global__",
        }

    def begin_row(self, *, timestamp_ns: int, evaluation_block: str) -> None:
        timestamp = int(timestamp_ns)
        block = str(evaluation_block)
        if self.timestamp_ns is not None and timestamp < self.timestamp_ns:
            raise CausalLeafHealthError("H4 external order regressed in feature timestamp")
        if self.timestamp_ns != timestamp:
            self.timestamp_ns = timestamp
            if block != self.active_block:
                if block in self.seen_blocks:
                    raise CausalLeafHealthError("H4 evaluation blocks are not contiguous and chronological")
                self.references = {
                    horizon: {
                        level: {key: state.copy() for key, state in by_key.items()}
                        for level, by_key in by_level.items()
                    }
                    for horizon, by_level in self.states.items()
                }
                self.active_block = block
                self.seen_blocks.add(block)
        elif block != self.active_block:
            raise CausalLeafHealthError("one H4 feature timestamp spans multiple evaluation blocks")
        assert self.references is not None

    def score(self, *, family: str, side: str, head: str) -> np.ndarray:
        assert self.references is not None
        keys = self._keys(family, side, head)
        output = np.full(len(covariance_feature_names("base_health__h4")), np.nan, dtype=np.float32)
        for horizon_index, horizon in enumerate(HORIZONS):
            current = {
                level: self.states[horizon][level].get(keys[level])
                for level in keys
            }
            frozen = {
                level: self.references[horizon][level].get(keys[level])
                for level in keys
            }
            now, reference, weights, support = _hierarchical_matrices(
                {level: value for level, value in current.items() if value is not None},
                {level: value for level, value in frozen.items() if value is not None},
                self.config,
            )
            start = horizon_index * 5
            if now is not None and reference is not None:
                output[start:start + 5] = np.asarray(_diagnostics(now, reference, self.config), dtype=np.float32)
            if horizon_index == 0:
                output[10:14] = np.asarray((*weights, support), dtype=np.float32)
        return output

    def update(self, *, family: str, side: str, head: str, timestamp_ns: int, values: np.ndarray) -> None:
        if not np.isfinite(values).all():
            return
        keys = self._keys(family, side, head)
        for horizon in HORIZONS:
            for level, key in keys.items():
                state = self.states[horizon][level].get(key)
                if state is None:
                    state = _EWMCovariance.empty(len(self.fields))
                    self.states[horizon][level][key] = state
                state.update(values, int(timestamp_ns), horizon, self.config.value_clip)


class _PendingCovarianceUpdates:
    """Defer an equal-timestamp H4 update without an unbounded Python list."""

    def __init__(
        self, *, fields: Sequence[str], path: Path, batch_rows: int, max_disk_bytes: int,
    ) -> None:
        self.fields = tuple(map(str, fields))
        self.path = path
        self.batch_rows = max(1, int(batch_rows))
        self.max_disk_bytes = int(max_disk_bytes)
        self.rows: list[tuple[object, ...]] = []
        self.writer: pq.ParquetWriter | None = None
        self.schema = pa.schema([
            pa.field("family", pa.string()), pa.field("side_name", pa.string()),
            pa.field("head_name", pa.string()), pa.field("timestamp_ns", pa.int64()),
            *[pa.field(name, pa.float64()) for name in self.fields],
        ])

    def _flush(self) -> None:
        if not self.rows:
            return
        columns = list(zip(*self.rows, strict=True))
        table = pa.Table.from_arrays(
            [pa.array(column, type=field.type) for column, field in zip(columns, self.schema, strict=True)],
            schema=self.schema,
        )
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.writer.write_table(table)
        self.rows.clear()
        _check_bounded_file(self.path, maximum_bytes=self.max_disk_bytes, label="same-timestamp H4 update spill")

    def add(self, *, family: str, side: str, head: str, timestamp_ns: int, values: np.ndarray) -> None:
        self.rows.append((family, side, head, int(timestamp_ns), *map(float, values)))
        if len(self.rows) >= self.batch_rows:
            self._flush()

    def apply(self, state: _StreamingCovarianceState) -> None:
        if self.writer is None:
            for family, side, head, timestamp_ns, *values in self.rows:
                state.update(
                    family=str(family), side=str(side), head=str(head), timestamp_ns=int(timestamp_ns),
                    values=np.asarray(values, dtype=np.float64),
                )
            self.rows.clear()
            return
        self._flush()
        self.writer.close()
        self.writer = None
        try:
            parquet = pq.ParquetFile(self.path)
            names = parquet.schema_arrow.names
            for batch in parquet.iter_batches(batch_size=self.batch_rows, columns=names):
                data = batch.to_pydict()
                for index in range(batch.num_rows):
                    state.update(
                        family=str(data["family"][index]), side=str(data["side_name"][index]),
                        head=str(data["head_name"][index]), timestamp_ns=int(data["timestamp_ns"][index]),
                        values=np.asarray([data[field][index] for field in self.fields], dtype=np.float64),
                    )
        finally:
            self.path.unlink(missing_ok=True)


def _covariance_fields(
    selected_state: Path, *, context_columns: Sequence[str], config: CausalLeafHealthConfig,
    connection: duckdb.DuckDBPyConnection,
) -> tuple[int, tuple[str, ...]]:
    """Select the exact all-finite H4 input surface without decoding rows."""

    keys = sorted(config.selected_covariance_families)
    if not keys:
        return 0, ()
    key_frame = pd.DataFrame(keys, columns=list(_FAMILY))
    connection.register("h4_selected_keys", key_frame)
    source = f"read_parquet({_literal(str(selected_state))})"
    rows = int(connection.execute(
        f"SELECT count(*) FROM {source} s INNER JOIN h4_selected_keys k USING ({', '.join(_FAMILY)})"
    ).fetchone()[0])
    if rows == 0:
        return 0, ()
    schema = set(pq.ParquetFile(selected_state).schema_arrow.names)
    missing = [str(name) for name in context_columns if str(name) not in schema]
    if missing:
        raise CausalLeafHealthError(f"selected H4 state is missing causal context fields: {missing}")
    checks = ", ".join(
        f"COALESCE(bool_and(s.{_identifier(str(field))} IS NOT NULL "
        f"AND isfinite(CAST(s.{_identifier(str(field))} AS DOUBLE))), FALSE) AS f{index}"
        for index, field in enumerate(context_columns)
    )
    flags = connection.execute(
        f"SELECT {checks} FROM {source} s INNER JOIN h4_selected_keys k USING ({', '.join(_FAMILY)})"
    ).fetchone()
    usable = tuple(
        str(field) for field, flag in zip(context_columns, flags, strict=True) if bool(flag)
    )[: int(config.covariance_max_fields)]
    return rows, usable


def _materialise_h4_bounded(
    selected_state: Path, covariance_path: Path, *, context_columns: Sequence[str],
    config: CausalLeafHealthConfig, work_dir: Path, memory_limit: str,
    temp_disk_limit: str, batch_rows: int,
) -> int:
    """External-sort selected H4 rows, then stream the exact covariance state.

    The physical sort is deliberately limited to frozen covariance families,
    not to every selected H3/H4/H5 state.  Equal feature timestamps are
    scored before their updates; unusually large timestamp groups spill a
    compact update record to disk rather than growing an unbounded list.
    """

    output_limit = _size_limit_bytes(temp_disk_limit)
    database = work_dir / "h4_order.duckdb"
    spill = work_dir / "h4_order_tmp"
    ordered = work_dir / "h4_ordered_selected.parquet"
    connection = duckdb.connect(database=str(database))
    try:
        connection.execute("PRAGMA threads=2")
        connection.execute(f"PRAGMA memory_limit={_literal(memory_limit)}")
        connection.execute(f"PRAGMA temp_directory={_literal(str(spill))}")
        connection.execute(f"SET max_temp_directory_size={_literal(temp_disk_limit)}")
        rows, fields = _covariance_fields(
            selected_state, context_columns=context_columns, config=config, connection=connection,
        )
        if rows == 0 or not fields:
            _empty_covariance(covariance_path)
            return 0
        projected = [
            *_IDENTITY, "feature_generation_ts", "feature_contract_sha256", "head_name",
            "rule_signature", "contribution_direction", "family_ensemble_tree_contribution", *fields,
        ]
        select = ", ".join(f"s.{_identifier(name)}" for name in projected)
        connection.execute(
            f"""
            COPY (
              WITH source AS (
                SELECT *, row_number() OVER () AS __source_order
                FROM read_parquet({_literal(str(selected_state))})
              )
              SELECT {select}
              FROM source s
              INNER JOIN h4_selected_keys k USING ({', '.join(_FAMILY)})
              -- Transports intentionally replay historical timestamps under
              -- independent outer lineage.  Keep each transport contiguous:
              -- H4 is a transport-local prequential state, never a shared
              -- history across duplicate OOF replay rows.
              ORDER BY s.transport, s.feature_generation_ts, s.candidate_id, s.head_name, s.__source_order
            ) TO {_literal(str(ordered))} (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        connection.close()
        shutil.rmtree(spill, ignore_errors=True)
        database.unlink(missing_ok=True)
    _check_bounded_file(ordered, maximum_bytes=output_limit, label="H4 selected external order")

    covariance_config = CausalLeafCovarianceConfig(
        min_reference_rows=int(config.covariance_min_reference_rows),
        max_fields_per_family=int(config.covariance_max_fields),
    )
    names = covariance_feature_names("base_health__h4")
    writer = _ParquetRows(covariance_path, flush_rows=min(_DIRECT_WRITER_ROWS, max(1, int(batch_rows))))
    state: _StreamingCovarianceState | None = None
    pending: _PendingCovarianceUpdates | None = None
    active_transport: str | None = None
    current_timestamp: int | None = None
    try:
        parquet = pq.ParquetFile(ordered)
        for batch in parquet.iter_batches(batch_size=min(16_384, max(1, int(batch_rows)))):
            data = batch.to_pydict()
            for index in range(batch.num_rows):
                transport = str(data["transport"][index])
                if transport != active_transport:
                    if pending is not None and state is not None:
                        pending.apply(state)
                    active_transport = transport
                    state = _StreamingCovarianceState(fields, covariance_config)
                    pending = _PendingCovarianceUpdates(
                        fields=fields, path=work_dir / "h4_same_timestamp_updates.parquet",
                        batch_rows=min(16_384, max(1, int(batch_rows))), max_disk_bytes=output_limit,
                    )
                    current_timestamp = None
                assert state is not None and pending is not None
                timestamp_ns = _utc_ns(data["feature_generation_ts"][index])
                if current_timestamp is not None and timestamp_ns != current_timestamp:
                    pending.apply(state)
                current_timestamp = timestamp_ns
                side = str(data["side_name"][index])
                head = str(data["head_name"][index])
                family = "|".join((
                    str(data["feature_contract_sha256"][index]), side, head,
                    str(data["rule_signature"][index]), str(data["contribution_direction"][index]),
                ))
                state.begin_row(
                    timestamp_ns=timestamp_ns,
                    evaluation_block=str(data["fold_id"][index]),
                )
                diagnostics = state.score(family=family, side=side, head=head)
                row = {
                    field: data[field][index]
                    for field in (*_IDENTITY, "head_name", "contribution_direction", "rule_signature", "family_ensemble_tree_contribution")
                }
                row.update({name: np.float32(value) for name, value in zip(names, diagnostics, strict=True)})
                writer.add(row)
                pending.add(
                    family=family, side=side, head=head, timestamp_ns=timestamp_ns,
                    values=np.asarray([data[field][index] for field in fields], dtype=np.float64),
                )
        if current_timestamp is not None and pending is not None and state is not None:
            pending.apply(state)
        written = writer.close(empty_columns=[
            *_IDENTITY, "head_name", "contribution_direction", "rule_signature",
            "family_ensemble_tree_contribution", *names,
        ])
    finally:
        # A raised covariance/input contract must not leave a misleading
        # same-timestamp source around for a later retry in this temp root.
        if pending is not None:
            pending.path.unlink(missing_ok=True)
        ordered.unlink(missing_ok=True)
    _check_bounded_file(covariance_path, maximum_bytes=output_limit, label="H4 covariance diagnostics")
    return int(written)


def _materialise_h5_bounded(
    selected_state: Path, relationship_path: Path, *, context_columns: Sequence[str],
    config: CausalLeafHealthConfig, work_dir: Path, memory_limit: str, temp_disk_limit: str,
) -> int:
    """Materialise H5 relationship rows with capped DuckDB SQL only."""

    pairs = _relationship_break_columns(context_columns)
    if not config.selected_relationship_families or not pairs:
        _empty_relationships(relationship_path)
        return 0
    schema = set(pq.ParquetFile(selected_state).schema_arrow.names)
    required = {"h5_selection_active", "h2_instability", "h1_economic_residual_bps", *[name for values in pairs.values() for name in values]}
    missing = sorted(required.difference(schema))
    if missing:
        raise CausalLeafHealthError(f"selected H5 state is missing relationship inputs: {missing}")
    database = work_dir / "h5_relationships.duckdb"
    spill = work_dir / "h5_relationships_tmp"
    connection = duckdb.connect(database=str(database))
    try:
        connection.execute("PRAGMA threads=2")
        connection.execute(f"PRAGMA memory_limit={_literal(memory_limit)}")
        connection.execute(f"PRAGMA temp_directory={_literal(str(spill))}")
        connection.execute(f"SET max_temp_directory_size={_literal(temp_disk_limit)}")
        key_frame = pd.DataFrame(sorted(config.selected_relationship_families), columns=list(_FAMILY))
        connection.register("h5_selected_keys", key_frame)
        instability = (
            f"CASE WHEN isfinite(CAST(s.{_identifier('h2_instability')} AS DOUBLE)) "
            f"THEN CAST(s.{_identifier('h2_instability')} AS DOUBLE) ELSE 0.0 END"
        )
        economic = (
            f"CASE WHEN isfinite(CAST(s.{_identifier('h1_economic_residual_bps')} AS DOUBLE)) "
            f"THEN CAST(s.{_identifier('h1_economic_residual_bps')} AS DOUBLE) ELSE 0.0 END"
        )
        weight = (
            f"abs(CAST(s.family_ensemble_tree_contribution AS DOUBLE)) "
            f"* (1.0 / (1.0 + greatest({instability}, 0.0))) "
            f"* (1.0 + abs({economic}) / 100.0)"
        )
        arms: list[str] = []
        identity = ", ".join(f"s.{_identifier(name)}" for name in _IDENTITY)
        base = (
            f"{identity}, s.head_name, s.contribution_direction, s.rule_signature, "
            "s.family_ensemble_tree_contribution"
        )
        for pair, fields in sorted(pairs.items()):
            finite = [
                f"CASE WHEN s.{_identifier(field)} IS NULL OR NOT isfinite(CAST(s.{_identifier(field)} AS DOUBLE)) "
                f"THEN -1e308 ELSE abs(CAST(s.{_identifier(field)} AS DOUBLE)) END"
                for field in fields
            ]
            absent = " AND ".join(
                f"(s.{_identifier(field)} IS NULL OR NOT isfinite(CAST(s.{_identifier(field)} AS DOUBLE)))"
                for field in fields
            )
            value = f"(CASE WHEN {absent} THEN NULL ELSE greatest({', '.join(finite)}) END)"
            arms.append(
                f"SELECT {base}, {_literal(pair)} AS relationship_pair, "
                f"CAST({value} AS FLOAT) AS relationship_break, "
                f"CAST(COALESCE({value} >= {float(config.relationship_break_threshold)}, FALSE) AS BOOLEAN) AS material_break, "
                f"CAST({weight} AS DOUBLE) AS portable_economic_weight "
                "FROM selected s"
            )
        connection.execute(
            f"""
            COPY (
              WITH selected AS (
                SELECT s.*
                FROM read_parquet({_literal(str(selected_state))}) s
                INNER JOIN h5_selected_keys k USING ({', '.join(_FAMILY)})
                WHERE s.h5_selection_active IS NULL OR CAST(s.h5_selection_active AS DOUBLE) <> 0.0
              )
              {' UNION ALL '.join(arms)}
            ) TO {_literal(str(relationship_path))} (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        written = int(connection.execute(
            f"SELECT count(*) FROM read_parquet({_literal(str(relationship_path))})"
        ).fetchone()[0])
    finally:
        connection.close()
        shutil.rmtree(spill, ignore_errors=True)
        database.unlink(missing_ok=True)
    _check_bounded_file(
        relationship_path, maximum_bytes=_size_limit_bytes(temp_disk_limit), label="H5 relationship diagnostics",
    )
    return int(written)


def _materialise_h4_h5_bounded(
    selected_state: Path, *, covariance_path: Path, relationship_path: Path,
    context_columns: Sequence[str], config: CausalLeafHealthConfig, memory_limit: str,
    temp_disk_limit: str, batch_rows: int,
) -> tuple[int, int, pd.DataFrame]:
    """Write schema-compatible H4/H5 artifacts without loading selected state.

    The only durable input is the capped selected-family state Parquet.  H4
    source ordering is explicit and bounded; H5 remains a projection over
    selected rows and never materialises a pandas contribution frame.
    """

    work_dir = selected_state.parent / "h4_h5_bounded_work"
    work_dir.mkdir(exist_ok=False)
    try:
        covariance_rows = _materialise_h4_bounded(
            selected_state, covariance_path, context_columns=context_columns, config=config,
            work_dir=work_dir, memory_limit=memory_limit, temp_disk_limit=temp_disk_limit,
            batch_rows=batch_rows,
        )
        relationship_rows = _materialise_h5_bounded(
            selected_state, relationship_path, context_columns=context_columns, config=config,
            work_dir=work_dir, memory_limit=memory_limit, temp_disk_limit=temp_disk_limit,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    field_audit = pd.DataFrame({
        "context_feature": list(context_columns),
        "used_for_covariance": [
            column in set(context_columns[: int(config.covariance_max_fields)])
            for column in context_columns
        ],
    })
    return int(covariance_rows), int(relationship_rows), field_audit


def _merge_periods(
    scope_results: Sequence[Mapping[str, Any]], *, output: Path,
) -> int:
    frames: list[pd.DataFrame] = []
    for result in scope_results:
        contract, side, head = result["key"]
        reverse = result["family_reverse"]
        for path in result["period_paths"]:
            frame = pd.read_parquet(path)
            if frame.empty:
                continue
            # Each scope retains an integer family code only while streaming;
            # materialised audit rows expose token-free signatures below from
            # the portable snapshot table, not a raw leaf/token.
            frame["feature_contract_sha256"] = contract
            frame["side_name"] = side
            frame["head_name"] = head
            pairs = [reverse[int(code)] or ("", "") for code in frame["family_code"].to_numpy(dtype=np.int32)]
            frame["rule_signature"] = [item[0] for item in pairs]
            frame["contribution_direction"] = [item[1] for item in pairs]
            frame = frame.drop(columns="family_code")
            rows = frame["rows"].to_numpy(dtype=np.float64)
            residual = frame["residual_sum"].to_numpy(dtype=np.float64)
            effect = residual / np.maximum(rows, 1.0)
            variance = np.maximum(
                frame["residual_sq_sum"].to_numpy(dtype=np.float64) / np.maximum(rows, 1.0) - effect * effect,
                0.0,
            )
            frame["mean_prediction"] = frame["prediction_sum"].to_numpy(dtype=np.float64) / np.maximum(rows, 1.0)
            frame["posterior_correctness_raw"] = frame["successes"].to_numpy(dtype=np.float64) / np.maximum(rows, 1.0)
            frame["calibration_residual"] = frame["calibration_sum"].to_numpy(dtype=np.float64) / np.maximum(rows, 1.0)
            frame["mean_net_bps"] = frame["net_sum"].to_numpy(dtype=np.float64) / np.maximum(rows, 1.0)
            frame["economic_residual_bps"] = effect
            frame["economic_residual_se_bps"] = np.where(rows >= 2.0, np.sqrt(variance / np.maximum(rows, 1.0)), np.nan)
            frame["false_positive_loss_bps"] = frame["false_positive_loss_sum"].to_numpy(dtype=np.float64) / np.maximum(rows, 1.0)
            frames.append(frame)
    if not frames:
        pd.DataFrame().to_parquet(output, index=False, compression="zstd")
        return 0
    merged = pd.concat(frames, ignore_index=True)
    merged.to_parquet(output, index=False, compression="zstd")
    return int(len(merged))


def _merge_portability(paths: Sequence[Path], *, output: Path) -> int:
    populated = [path for path in paths if path.is_file() and pq.ParquetFile(path).metadata.num_rows > 0]
    if not populated:
        pd.DataFrame().to_parquet(output, index=False, compression="zstd")
        return 0
    frame = pd.concat([pd.read_parquet(path) for path in populated], ignore_index=True)
    frame.to_parquet(output, index=False, compression="zstd")
    return int(len(frame))


def _candidate_paths(store: StrictEventStore) -> str:
    index = _scope_rows(store, "candidate")
    return _paths_literal(store.root, index["path"].astype(str))


def _final_candidate_partitions(
    store: StrictEventStore,
) -> tuple[list[tuple[str, str, str]], dict[tuple[str, str], tuple[Path, ...]]]:
    """Return bounded final-output partitions from immutable candidate parts.

    ``transport`` is deliberately retained as a data column rather than a
    physical event-store partition.  Reading that one dictionary column is
    cheap and avoids a global candidate ``DISTINCT`` merely to discover the
    output groups.  The physical candidate layout already fixes decision
    month and meta partition, so every returned unit is one
    ``(transport, meta_partition, decision_month)`` slice.  Concatenating
    those units in lexical key order is exactly the final health ordering
    ``transport, meta_partition, decision_ts, candidate_id`` because months
    are ISO ordered and each unit is sorted by its final in-month keys.
    """

    index = _scope_rows(store, "candidate").drop_duplicates("path").copy()
    required = {"path", "month", "meta_partition"}
    missing = sorted(required.difference(index.columns))
    if missing:
        raise CausalLeafHealthError(f"strict event-store candidate index lacks final partition fields: {missing}")
    paths_by_month_partition: dict[tuple[str, str], list[Path]] = defaultdict(list)
    partitions: set[tuple[str, str, str]] = set()
    for row in index.sort_values(["meta_partition", "month", "path"], kind="stable").itertuples(index=False):
        month = str(row.month)
        meta_partition = str(row.meta_partition)
        if not re.fullmatch(r"\d{4}-\d{2}", month):
            raise CausalLeafHealthError(f"invalid decision month in strict event-store index: {month!r}")
        if not meta_partition:
            raise CausalLeafHealthError("blank strict meta partition in event-store candidate index")
        path = store.root / str(row.path)
        if not path.is_file():
            raise CausalLeafHealthError(f"strict event-store candidate part is missing: {path}")
        paths_by_month_partition[(meta_partition, month)].append(path)
        parquet = pq.ParquetFile(path)
        if "transport" not in parquet.schema_arrow.names:
            raise CausalLeafHealthError(f"strict event-store candidate part lacks transport: {path}")
        # The event-store writer caps physical files.  Decode one dictionary
        # column at a time instead of materialising an all-candidate pandas
        # frame solely to discover transport slices.
        for batch in parquet.iter_batches(batch_size=_DIRECT_WRITER_ROWS, columns=["transport"]):
            for transport in batch.column(0).to_pylist():
                value = str(transport)
                if transport is None or not value:
                    raise CausalLeafHealthError("strict event-store candidate part has blank transport")
                partitions.add((value, meta_partition, month))
    if not partitions:
        raise CausalLeafHealthError("strict event-store has no candidate final-output partitions")
    return sorted(partitions), {
        key: tuple(values) for key, values in paths_by_month_partition.items()
    }


def _final_month_bounds(month: str) -> tuple[str, str]:
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    return start.isoformat(), end.isoformat()


def _final_partition_predicate(
    *, transport: str, meta_partition: str, start: str, end: str, decision_type: pa.DataType,
) -> str:
    """Build a type-preserving final-health source predicate.

    Strict production artifacts use UTC timestamps, while compact fixtures and
    legacy event-store states may retain their UTC nanoseconds as integers.
    Candidate, direct and selected-H4/H5 artifacts are allowed to use either
    representation independently, provided their equality join remains valid.
    Filtering every source in its native representation avoids a broad cast
    over a Parquet scan and keeps the month partition bounded.
    """

    if pa.types.is_timestamp(decision_type):
        time_predicate = (
            f"decision_ts >= {_literal(start)}::TIMESTAMPTZ "
            f"AND decision_ts < {_literal(end)}::TIMESTAMPTZ"
        )
    elif pa.types.is_integer(decision_type):
        time_predicate = (
            f"decision_ts >= {int(pd.Timestamp(start).value)} "
            f"AND decision_ts < {int(pd.Timestamp(end).value)}"
        )
    else:
        raise CausalLeafHealthError(
            f"unsupported strict final-health decision_ts type: {decision_type}"
        )
    return (
        f"transport={_literal(transport)} AND meta_partition={_literal(meta_partition)} "
        f"AND {time_predicate}"
    )


def _decision_type(path: Path, *, label: str) -> pa.DataType:
    schema = pq.ParquetFile(path).schema_arrow
    if "decision_ts" not in schema.names:
        raise CausalLeafHealthError(f"{label} lacks decision_ts: {path}")
    return schema.field("decision_ts").type


def _append_final_health_part(
    source: Path,
    *, output: Path,
    writer: pq.ParquetWriter | None,
    schema: pa.Schema | None,
) -> tuple[pq.ParquetWriter, pa.Schema, int]:
    """Append one already-sorted final-health part without a whole-frame read."""

    parquet = pq.ParquetFile(source)
    part_schema = parquet.schema_arrow
    if writer is None:
        writer = pq.ParquetWriter(output, part_schema, compression="zstd")
        schema = part_schema
    elif schema is None or not schema.equals(part_schema, check_metadata=True):
        raise CausalLeafHealthError("partitioned final health output schema drifted between candidate slices")
    rows = 0
    for batch in parquet.iter_batches(batch_size=_DIRECT_WRITER_ROWS):
        # Preserve the DuckDB-emitted field types/order exactly.  Arrow does
        # not need to decode the entire part to append it to the final file.
        writer.write_table(pa.Table.from_batches([batch], schema=part_schema))
        rows += int(batch.num_rows)
    assert schema is not None
    return writer, schema, int(rows)


def _final_health(
    *, store: StrictEventStore, direct_paths: Sequence[Path], h4_path: Path, h4_names: Sequence[str],
    h5_path: Path, h5_names: Sequence[str], output: Path, memory_limit: str, temp_disk_limit: str, temp_dir: Path,
) -> int:
    """Merge H1--H5 in bounded candidate slices, preserving final semantics.

    The old plan made a single global ``DISTINCT candidates`` relation and a
    global direct-health aggregate before joining H4/H5.  On the sealed
    7.3M-row store that relation spilled 7.4 GiB and exhausted the declared
    8 GiB temporary budget.  This plan intentionally preserves the same
    joins, zero fills, aggregations and SQL output order, but performs them
    independently for immutable decision-month/transport/meta partitions.
    Only one part is open in DuckDB at a time; Arrow then appends that sorted
    part to the final Parquet without a global re-sort or dataframe decode.
    """

    direct = [path for path in direct_paths if path.is_file() and pq.ParquetFile(path).metadata.num_rows > 0]
    if not direct:
        raise CausalLeafHealthError("incremental health materialisation produced no direct scope rows")
    if not h4_path.is_file() or not h5_path.is_file():
        raise CausalLeafHealthError("incremental health materialisation lacks its H4 or H5 candidate artifact")
    partitions, candidate_paths_by_month_partition = _final_candidate_partitions(store)
    temp_dir.mkdir(parents=True, exist_ok=True)
    work_dir = temp_dir / "final_health_parts"
    work_dir.mkdir(exist_ok=False)
    direct_paths_literal = "[" + ", ".join(_literal(str(path)) for path in direct) + "]"
    present = set().union(*(set(pq.ParquetFile(path).schema_arrow.names) for path in direct))
    direct_types = {str(_decision_type(path, label="direct health part")) for path in direct}
    if len(direct_types) != 1:
        raise CausalLeafHealthError("direct health parts disagree on their decision_ts representation")
    direct_decision_type = _decision_type(direct[0], label="direct health part")
    h4_decision_type = _decision_type(h4_path, label="H4 candidate artifact")
    h5_decision_type = _decision_type(h5_path, label="H5 candidate artifact")
    aggregate = ", ".join(
        (
            f"CAST(COALESCE(sum(d.\"{name}\"), 0.0) AS FLOAT) AS \"{name}\""
            if name in present else f"CAST(0.0 AS FLOAT) AS \"{name}\""
        )
        for name in _DIRECT_NAMES
    )
    entropy_aggregate = ", ".join(
        f"CAST(COALESCE(sum(d.\"{name}\"), 0.0) AS DOUBLE) AS \"{name}\""
        if name in present else f"CAST(0.0 AS DOUBLE) AS \"{name}\""
        for name in (_ENTROPY_MASS, _ENTROPY_MASS_LOG_MASS)
    )
    selects = [f"b.{field}" for field in _IDENTITY]
    selects.extend(f"COALESCE(d.\"{name}\", 0.0)::FLOAT AS \"{name}\"" for name in _DIRECT_NAMES)
    for name in h4_names:
        if name.endswith("__active_abs_contribution"):
            h1_name = name.replace("base_health__h4__", "base_health__h1__", 1)
            selects.append(f"COALESCE(d.\"{h1_name}\", 0.0)::FLOAT AS \"{name}\"")
        else:
            selects.append(f"COALESCE(h4.\"{name}\", 0.0)::FLOAT AS \"{name}\"")
    selects.extend(f"COALESCE(h5.\"{name}\", 0.0)::FLOAT AS \"{name}\"" for name in h5_names)
    # Shannon entropy of the absolute contribution distribution.  The
    # probability normalisation is over all heads/families that share the
    # complete strict candidate identity.  A zero-mass candidate has no
    # distribution, and is explicitly assigned zero rather than NaN.
    selects.append(
        f"CASE WHEN COALESCE(d.\"{_ENTROPY_MASS}\", 0.0) <= 0.0 THEN 0.0::FLOAT "
        f"ELSE GREATEST(0.0, ln(d.\"{_ENTROPY_MASS}\") - "
        f"d.\"{_ENTROPY_MASS_LOG_MASS}\" / d.\"{_ENTROPY_MASS}\")::FLOAT END "
        f"AS \"{REASONING_ENTROPY_FIELD}\""
    )
    writer: pq.ParquetWriter | None = None
    final_schema: pa.Schema | None = None
    written = 0
    try:
        for index, (transport, meta_partition, month) in enumerate(partitions):
            start, end = _final_month_bounds(month)
            candidate_paths = candidate_paths_by_month_partition.get((meta_partition, month), ())
            if not candidate_paths:
                raise CausalLeafHealthError("final-health candidate partition lacks immutable source parts")
            candidate_predicate = _final_partition_predicate(
                transport=transport, meta_partition=meta_partition, start=start, end=end,
                decision_type=_decision_type(candidate_paths[0], label="strict candidate part"),
            )
            direct_predicate = _final_partition_predicate(
                transport=transport, meta_partition=meta_partition, start=start, end=end,
                decision_type=direct_decision_type,
            )
            h4_predicate = _final_partition_predicate(
                transport=transport, meta_partition=meta_partition, start=start, end=end,
                decision_type=h4_decision_type,
            )
            h5_predicate = _final_partition_predicate(
                transport=transport, meta_partition=meta_partition, start=start, end=end,
                decision_type=h5_decision_type,
            )
            candidate_paths_literal = "[" + ", ".join(_literal(str(path)) for path in candidate_paths) + "]"
            part_path = work_dir / f"part_{index:05d}.parquet"
            database = work_dir / f"part_{index:05d}.duckdb"
            spill = work_dir / f"part_{index:05d}_tmp"
            connection = duckdb.connect(database=str(database))
            try:
                connection.execute("PRAGMA threads=2")
                connection.execute(f"PRAGMA memory_limit={_literal(memory_limit)}")
                connection.execute(f"PRAGMA temp_directory={_literal(str(spill))}")
                connection.execute(f"SET max_temp_directory_size={_literal(temp_disk_limit)}")
                connection.execute(
                    f"CREATE TEMP VIEW direct_health AS SELECT {', '.join(_IDENTITY)}, {aggregate}, {entropy_aggregate} "
                    f"FROM read_parquet({direct_paths_literal}, union_by_name=true) d "
                    f"WHERE {direct_predicate} GROUP BY {', '.join(_IDENTITY)}"
                )
                connection.execute(
                    f"COPY (SELECT {', '.join(selects)} FROM "
                    f"(SELECT DISTINCT {', '.join(_IDENTITY)} FROM "
                    f"read_parquet({candidate_paths_literal}, hive_partitioning=false) WHERE {candidate_predicate}) b "
                    f"LEFT JOIN direct_health d USING ({', '.join(_IDENTITY)}) "
                    f"LEFT JOIN (SELECT * FROM read_parquet({_literal(str(h4_path))}) WHERE {h4_predicate}) h4 "
                    f"USING ({', '.join(_IDENTITY)}) "
                    f"LEFT JOIN (SELECT * FROM read_parquet({_literal(str(h5_path))}) WHERE {h5_predicate}) h5 "
                    f"USING ({', '.join(_IDENTITY)}) "
                    f"ORDER BY b.decision_ts, b.candidate_id) "
                    f"TO {_literal(str(part_path))} (FORMAT PARQUET, COMPRESSION ZSTD)"
                )
            finally:
                connection.close()
                shutil.rmtree(spill, ignore_errors=True)
                database.unlink(missing_ok=True)
            _check_bounded_file(
                part_path, maximum_bytes=_size_limit_bytes(temp_disk_limit), label="final health candidate partition",
            )
            writer, final_schema, rows = _append_final_health_part(
                part_path, output=output, writer=writer, schema=final_schema,
            )
            written += int(rows)
            part_path.unlink(missing_ok=True)
        if writer is None or written <= 0:
            raise CausalLeafHealthError("partitioned final-health materialisation emitted no candidate rows")
        return int(written)
    finally:
        if writer is not None:
            writer.close()
        shutil.rmtree(work_dir, ignore_errors=True)


def _write_reasoning_entropy_coverage_audit(
    *, health_path: Path, direct_paths: Sequence[Path], output: Path,
    memory_limit: str, temp_disk_limit: str, temp_dir: Path,
) -> int:
    """Write a compact coverage/provenance audit for the S2 entropy field.

    The query scans only the already-written narrow direct scope parts and the
    final candidate table.  It never materialises per-family rows or joins
    realised outcomes.  The audit makes zero total contribution mass visible
    separately from a valid concentrated distribution whose entropy is zero.
    """

    direct = [path for path in direct_paths if path.is_file() and pq.ParquetFile(path).metadata.num_rows > 0]
    if not direct:
        raise CausalLeafHealthError("reasoning entropy audit lacks direct contribution parts")
    direct_paths_literal = "[" + ", ".join(_literal(str(path)) for path in direct) + "]"
    database = temp_dir / "reasoning_entropy_audit.duckdb"
    spill = temp_dir / "reasoning_entropy_audit_tmp"
    connection = duckdb.connect(database=str(database))
    try:
        connection.execute("PRAGMA threads=2")
        connection.execute(f"PRAGMA memory_limit={_literal(memory_limit)}")
        connection.execute(f"PRAGMA temp_directory={_literal(str(spill))}")
        connection.execute(f"SET max_temp_directory_size={_literal(temp_disk_limit)}")
        identity = ", ".join(_IDENTITY)
        connection.execute(
            f"""
            COPY (
              WITH direct_mass AS (
                SELECT {identity}, sum(\"{_ENTROPY_MASS}\") AS total_abs_mass
                FROM read_parquet({direct_paths_literal}, union_by_name=true)
                GROUP BY {identity}
              )
              SELECT h.transport, h.meta_partition, h.side_name,
                count(*)::BIGINT AS candidate_rows,
                sum(CASE WHEN COALESCE(d.total_abs_mass, 0.0) <= 0.0 THEN 1 ELSE 0 END)::BIGINT
                  AS zero_total_mass_rows,
                avg(COALESCE(d.total_abs_mass, 0.0))::DOUBLE AS mean_total_abs_mass,
                min(h.\"{REASONING_ENTROPY_FIELD}\")::DOUBLE AS min_entropy,
                max(h.\"{REASONING_ENTROPY_FIELD}\")::DOUBLE AS max_entropy,
                avg(h.\"{REASONING_ENTROPY_FIELD}\")::DOUBLE AS mean_entropy
              FROM read_parquet({_literal(str(health_path))}) h
              LEFT JOIN direct_mass d USING ({identity})
              GROUP BY h.transport, h.meta_partition, h.side_name
              ORDER BY h.transport, h.meta_partition, h.side_name
            ) TO {_literal(str(output))} (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        rows = int(connection.execute(
            f"SELECT count(*) FROM read_parquet({_literal(str(output))})"
        ).fetchone()[0])
        return rows
    finally:
        connection.close()
        shutil.rmtree(spill, ignore_errors=True)
        database.unlink(missing_ok=True)


def materialize_strict_oof_causal_leaf_health_event_incremental(
    event_store: StrictEventStore | str | Path,
    contribution_event_streams: StrictContributionEventStreams | str | Path,
    output_dir: str | Path, *, causal_context: pd.DataFrame, context_feature_columns: Sequence[str],
    config: CausalLeafHealthConfig = CausalLeafHealthConfig(), batch_rows: int = 131_072,
    max_open_parts: int = 8, max_selected_state_rows: int = _SELECTED_STATE_LIMIT_DEFAULT,
    memory_limit: str = "2GB", temp_disk_limit: str = "8GB", verify_event_store_parts: bool = False,
    verify_stream_parts: bool = False,
) -> Path:
    """Materialise strict H1--H5 with a bounded contribution event pass.

    The sidecar and event store must share the exact sealed manifest.  No
    target/model selection is performed here; this function is only the
    strict-prequential feature boundary used by later fixed ablations.
    """

    started = time.monotonic()
    config.validate()
    if int(batch_rows) <= 0 or int(max_open_parts) <= 0 or int(max_selected_state_rows) <= 0:
        raise CausalLeafHealthError("incremental event health limits must be positive")
    store = event_store if isinstance(event_store, StrictEventStore) else load_strict_event_store(
        event_store, verify_parts=verify_event_store_parts, verify_source=True,
    )
    streams = contribution_event_streams if isinstance(contribution_event_streams, StrictContributionEventStreams) else load_strict_contribution_event_streams(
        contribution_event_streams, verify_parts=verify_stream_parts,
    )
    declared = str(streams.manifest.get("source_event_store_manifest_sha256", ""))
    if declared != _hash(store.manifest_path):
        raise CausalLeafHealthError("contribution-event sidecar does not derive from this sealed event store")
    context_times, context_values, context_columns = _context_arrays(causal_context, context_feature_columns, config)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite causal leaf health artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        global_prior, global_groups = _prior_series(
            store, temporary, memory_limit=memory_limit, temp_disk_limit=temp_disk_limit,
        )
        index = streams.part_index.loc[streams.part_index["dataset"].eq("feature_event_order")]
        keys = [tuple(map(str, item)) for item in index.loc[:, list(_SCOPE)].drop_duplicates().sort_values(list(_SCOPE), kind="stable").itertuples(index=False, name=None)]
        if not keys:
            raise CausalLeafHealthError("contribution-event sidecar has no feature-time scopes")
        scope_results: list[dict[str, Any]] = []
        selected_total = 0
        for number, key in enumerate(keys):
            scope_dir = temporary / f"scope_{number:03d}"
            scope_dir.mkdir()
            remaining = int(max_selected_state_rows) - selected_total
            if remaining <= 0:
                raise CausalLeafHealthError("selected H3/H4/H5 state audit exceeds max_selected_state_rows")
            result = _process_scope(
                streams, key=key, output_dir=scope_dir, global_prior=global_prior,
                context_times=context_times, context_values=context_values, context_columns=context_columns,
                config=config, batch_rows=int(batch_rows), max_open_parts=int(max_open_parts),
                max_selected_state_rows=remaining,
            )
            selected_total += int(result["selected_rows"])
            scope_results.append(result)
        selected_state = temporary / "base_leaf_family_candidate_states.parquet"
        selected_rows = _merge_selected(
            [item["selected_path"] for item in scope_results], output=selected_state,
            context_columns=context_columns, memory_limit=memory_limit,
            temp_disk_limit=temp_disk_limit,
        )
        # H4/H5 are source-filtered to frozen selected state rows.  The
        # selected state is capped but can still be multi-gigabyte once ten
        # context fields are present, so this postprocess must not decode it
        # into one pandas frame.
        covariance_path = temporary / "leaf_covariance_diagnostics.parquet"
        relationship_path = temporary / "leaf_relationship_breaks.parquet"
        covariance_rows, relationship_rows, field_audit = _materialise_h4_h5_bounded(
            selected_state, covariance_path=covariance_path, relationship_path=relationship_path,
            context_columns=context_columns, config=config, memory_limit=memory_limit,
            temp_disk_limit=temp_disk_limit, batch_rows=int(batch_rows),
        )
        h4_cells = temporary / "h4_cells.parquet"
        h4_metrics, _ = _write_h4_cells(
            selected_state, covariance_path,
            [item["denominator_path"] for item in scope_results if int(item["denominator_rows"]) > 0], h4_cells,
            memory_limit=memory_limit, temp_disk_limit=temp_disk_limit,
            temp_dir=temporary / "h4_cells_tmp",
        )
        h4_path = temporary / "h4.parquet"
        h5_path = temporary / "h5.parquet"
        merge = duckdb.connect(database=str(temporary / "postprocess.duckdb"))
        try:
            merge.execute("PRAGMA threads=2")
            merge.execute(f"PRAGMA memory_limit={_literal(memory_limit)}")
            merge.execute(f"PRAGMA temp_directory={_literal(str(temporary / 'postprocess_tmp'))}")
            merge.execute(f"SET max_temp_directory_size={_literal(temp_disk_limit)}")
            h4_partial = _pivot_partial(merge, input_path=h4_cells, section="h4", metrics=(*h4_metrics, "availability"), output=h4_path)
            h4_names: list[str] = []
            width = len(h4_metrics) + 1
            for index, (head, direction) in enumerate((pair for head in HEADS for pair in ((head, "positive"), (head, "negative")))):
                start = index * width
                h4_names.extend(h4_partial[start:start + width])
                h4_names.append(f"base_health__h4__{head}__{direction}__active_abs_contribution")
            pairs = sorted(_relationship_break_columns(context_columns))
            if relationship_rows == 0:
                empty = temporary / "empty_h5_candidate.parquet"
                pd.DataFrame(columns=list(_IDENTITY)).to_parquet(empty, index=False, compression="zstd")
                _copy_h5(merge, candidate_view=f"read_parquet({_literal(str(empty))})", relationship_path=relationship_path, output=h5_path, pairs=pairs)
            else:
                merge.execute(f"CREATE TEMP VIEW h5_candidates AS SELECT DISTINCT {', '.join(_IDENTITY)} FROM read_parquet({_literal(str(relationship_path))})")
                _copy_h5(merge, candidate_view="h5_candidates", relationship_path=relationship_path, output=h5_path, pairs=pairs)
            h5_names = tuple(name for name in pq.ParquetFile(h5_path).schema_arrow.names if name not in _IDENTITY)
        finally:
            merge.close()
            shutil.rmtree(temporary / "postprocess_tmp", ignore_errors=True)
            (temporary / "postprocess.duckdb").unlink(missing_ok=True)
        health_path = temporary / "base_leaf_health_features_oof.parquet"
        health_rows = _final_health(
            store=store, direct_paths=[item["direct_path"] for item in scope_results],
            h4_path=h4_path, h4_names=tuple(h4_names), h5_path=h5_path, h5_names=h5_names,
            output=health_path, memory_limit=memory_limit, temp_disk_limit=temp_disk_limit, temp_dir=temporary,
        )
        entropy_audit_path = temporary / "reasoning_entropy_coverage_audit.parquet"
        entropy_audit_rows = _write_reasoning_entropy_coverage_audit(
            health_path=health_path, direct_paths=[item["direct_path"] for item in scope_results],
            output=entropy_audit_path, memory_limit=memory_limit, temp_disk_limit=temp_disk_limit,
            temp_dir=temporary,
        )
        period_path = temporary / "leaf_period_metrics.parquet"
        period_rows = _merge_periods(scope_results, output=period_path)
        portability_path = temporary / "leaf_portability_scores.parquet"
        portability_rows = _merge_portability([item["portability_path"] for item in scope_results], output=portability_path)
        explain_path = temporary / "covariance_explainability.parquet"
        pd.DataFrame({
            "status": ["NOT_FITTED_IN_STATE_MATERIALISATION"],
            "reason": ["C5/C6 held-out explanatory regressions are later transport ablations"],
            "uses_outcomes": [False],
        }).to_parquet(explain_path, index=False, compression="zstd")
        files = (selected_state, health_path, entropy_audit_path, period_path, portability_path, covariance_path, relationship_path, explain_path)
        manifest = {
            "schema": SCHEMA, "status": STATUS, "created_utc": datetime.now(timezone.utc).isoformat(),
            "strict_roots": list(store.manifest["source"]["strict_roots"]),
            "strict_root_manifest_sha256": dict(store.manifest["source"]["strict_root_manifest_sha256"]),
            "strict_event_store_manifest_sha256": _hash(store.manifest_path),
            "strict_contribution_event_stream_manifest_sha256": _hash(streams.manifest_path),
            "contract": {
                "family_identity": "feature_contract_sha256, side, head, rule_signature, contribution_direction",
                "raw_leaf_ids": "rejected; only token-free rule-family signatures are present",
                "history": "only label_available_ts < feature_generation_ts; complete same-time feature blocks score before resolution",
                "state_engine": "event-incremental NumPy/Numba H1 with month-frozen H2/H3 and selected-family H4/H5",
                "scope_plan": "one decision-month sidecar window per scope; no scope-wide contribution sort/join/spool",
                "covariance": "H4 source-filtered to frozen selected families; full H1 contribution mass remains the denominator",
                "reasoning_entropy": "candidate-level Shannon entropy over abs current family-tree contributions; mass is normalised over the complete strict identity across semantic heads and zero total mass maps to zero",
            },
            "config": _config_payload(config), "context_columns": list(context_columns),
            "covariance_field_audit": field_audit.to_dict("records"),
            "row_counts": {
                "family_candidate_states": selected_rows, "health_features": health_rows,
                "reasoning_entropy_coverage_rows": int(entropy_audit_rows),
                "period_metrics": period_rows, "portability_scores": portability_rows,
                "covariance_diagnostics": int(covariance_rows), "relationship_breaks": int(relationship_rows),
                "global_h1_resolution_groups": global_groups, "scopes": len(scope_results),
                "families_by_scope": {"|".join(item["key"]): int(item["family_count"]) for item in scope_results},
            },
            "performance": {
                "elapsed_seconds": round(time.monotonic() - started, 3), "batch_rows": int(batch_rows),
                "max_open_parts": int(max_open_parts), "memory_limit": memory_limit,
                "temp_disk_limit": temp_disk_limit, "global_contribution_join": False,
                "scope_contribution_sort": False, "full_event_store_part_checksum_audit": bool(verify_event_store_parts),
            },
            "sha256": {path.name: _hash(path) for path in files},
        }
        (temporary / "leaf_covariance_reference_manifest.json").write_text(json.dumps({
            "schema": SCHEMA, "status": "CAUSAL_COVARIANCE_REFERENCE_DECLARED",
            "contract": manifest["contract"]["covariance"], "context_columns": list(context_columns),
            "covariance_field_audit": manifest["covariance_field_audit"],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "leaf_failure_classification.yaml").write_text(json.dumps({
            "schema": SCHEMA, "classification_counts": {},
            "labels": ["STABLE_PORTABLE", "SUPPORT_SHIFT_ONLY", "CALIBRATION_DRIFT", "COMPOSITION_DRIFT", "REGIME_CONDITIONAL", "COVARIANCE_CONDITIONAL", "GLOBAL_PERIOD_SENSITIVE", "UNEXPLAINED_CONCEPT_BREAK", "LOW_SUPPORT_UNCERTAIN", "META_HARMFUL"],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary / "health_materialization_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        return target
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = ["materialize_strict_oof_causal_leaf_health_event_incremental"]
