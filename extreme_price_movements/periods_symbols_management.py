"""Leakage-safe time/symbol slice planner for event-based trading ML.

Non-negotiable rules enforced by design:
1. Split membership is assigned by ``t0`` only.
2. Purging is based on overlap of ``[t0, t1]`` intervals.
3. Outer test is strictly later than outer train.
4. Outer test is never used for tuning.
5. Embargo is applied after the test interval anchor.
6. Meta model fitting uses only outer-train rows with OOF base coverage.
7. Ridge sizer fitting uses only outer-train rows with OOF upstream coverage.
8. Final outer-test predictions come from models refit on full outer-train.
9. Time holdout and symbol holdout are separate and composable.
10. Backtest consumers receive contiguous outer-test blocks.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Optional

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

CONSUMER_ROLES: tuple[str, ...] = (
    "regime_search",
    "barrier_search",
    "base_model_fit",
    "meta_model_fit",
    "ridge_sizer_fit",
    "utility_policy_tuning",
    "policy_optimiser",
    "backtest_eval",
    "full_inference_fit",
)

DEDUP_QUOTES: tuple[str, ...] = ("USDT", "USDC", "BUSD")
DEDUP_QUOTE_PRIORITY: dict[str, int] = {
    "USDT": 0,
    "USDC": 1,
    "BUSD": 2,
}


@dataclass(frozen=True)
class EventSchema:
    event_id_col: str = "event_id"
    symbol_col: str = "symbol"
    t0_col: str = "t0"
    t1_col: str = "t1"
    group_cols: tuple[str, ...] = ()


@dataclass(frozen=True)
class PurgePolicy:
    purge_on_overlap: bool = True
    embargo_timedelta: pd.Timedelta = pd.Timedelta("0s")
    max_lookback_timedelta: Optional[pd.Timedelta] = None
    max_holding_timedelta: Optional[pd.Timedelta] = None
    purge_with_test_interval_end: bool = True
    purge_scope: Literal["global", "same_symbol", "same_group"] = "global"
    purge_group_col: Optional[str] = None


@dataclass(frozen=True)
class SymbolPolicy:
    mode: Literal[
        "all_symbols",
        "subset_symbols",
        "disjoint_symbol_holdout",
        "group_holdout",
        "fixed_symbol_partition",
    ] = "all_symbols"
    group_col: Optional[str] = None
    train_symbols: Optional[tuple[str, ...]] = None
    valid_symbols: Optional[tuple[str, ...]] = None
    test_symbols: Optional[tuple[str, ...]] = None
    subset_fraction: Optional[float] = None
    min_symbols_per_split: int = 1
    unseen_test_symbols: bool = True
    random_state: int = 42


@dataclass(frozen=True)
class SamplingPolicy:
    mode: Literal["full", "subsample"] = "full"
    event_fraction: float = 1.0
    symbol_fraction: float = 1.0
    max_events: Optional[int] = None
    max_symbols: Optional[int] = None
    random_state: int = 42
    event_sampling_method: Literal[
        "none", "head", "tail", "time_block", "every_kth"
    ] = "none"
    sample_train: bool = True
    sample_inner_train: bool = True
    sample_valid: bool = False
    sample_inner_valid: bool = False
    sample_test: bool = False
    sample_symbols_on_train: bool = True
    sample_symbols_on_inner_train: bool = True
    sample_symbols_on_valid: bool = False
    sample_symbols_on_inner_valid: bool = False
    sample_symbols_on_test: bool = False


@dataclass(frozen=True)
class OuterFoldConfig:
    train_mode: Literal["rolling", "expanding"] = "expanding"
    train_span: pd.Timedelta = pd.Timedelta("365D")
    valid_span: Optional[pd.Timedelta] = None
    test_span: pd.Timedelta = pd.Timedelta("30D")
    step_span: pd.Timedelta = pd.Timedelta("30D")
    n_folds: Optional[int] = None


@dataclass(frozen=True)
class InnerFoldConfig:
    n_splits: int = 4
    use_purging: bool = True
    use_embargo: bool = True


@dataclass(frozen=True)
class PlannerPresetConfig:
    preset_name: Literal["robust", "fast"]
    outer: OuterFoldConfig
    inner: InnerFoldConfig
    sampling: SamplingPolicy
    symbol_policy: SymbolPolicy
    purge_policy: PurgePolicy


@dataclass(frozen=True)
class SlicePlannerConfig:
    schema: EventSchema = EventSchema()
    preset: PlannerPresetConfig = PlannerPresetConfig(
        preset_name="robust",
        outer=OuterFoldConfig(),
        inner=InnerFoldConfig(),
        sampling=SamplingPolicy(),
        symbol_policy=SymbolPolicy(),
        purge_policy=PurgePolicy(),
    )
    consumer_overrides: dict[str, Any] = field(default_factory=dict)
    dtype_optimisation: bool = True
    cast_symbol_to_category: bool = True
    cast_group_cols_to_category: bool = True
    float_downcast: bool = True
    int_downcast: bool = True
    downcast_exclude_cols: tuple[str, ...] = ()
    timezone: Optional[str] = "UTC"
    min_rows_per_fold: int = 100
    min_symbols_per_fold: int = 5
    policy_optimiser_holdout_frac: float = 0.10
    strict_validation: bool = True
    silent: bool = False
    tprint: Optional[Callable[[str], None]] = None

    @classmethod
    def robust_defaults(
        cls, *, schema: EventSchema = EventSchema(), timezone: Optional[str] = "UTC"
    ) -> "SlicePlannerConfig":
        return cls(
            schema=schema,
            preset=PlannerPresetConfig(
                preset_name="robust",
                outer=OuterFoldConfig(
                    train_mode="expanding",
                    train_span=pd.Timedelta("365D"),
                    valid_span=pd.Timedelta("30D"),
                    test_span=pd.Timedelta("30D"),
                    step_span=pd.Timedelta("30D"),
                ),
                inner=InnerFoldConfig(n_splits=4, use_purging=True, use_embargo=True),
                sampling=SamplingPolicy(
                    mode="full",
                    event_fraction=1.0,
                    symbol_fraction=1.0,
                    event_sampling_method="none",
                ),
                symbol_policy=SymbolPolicy(mode="all_symbols", min_symbols_per_split=8),
                purge_policy=PurgePolicy(
                    embargo_timedelta=pd.Timedelta("1D"),
                    purge_scope="global",
                ),
            ),
            timezone=timezone,
            strict_validation=True,
            min_rows_per_fold=50,
            min_symbols_per_fold=8,
        )

    @classmethod
    def fast_defaults(
        cls, *, schema: EventSchema = EventSchema(), timezone: Optional[str] = "UTC"
    ) -> "SlicePlannerConfig":
        return cls(
            schema=schema,
            preset=PlannerPresetConfig(
                preset_name="fast",
                outer=OuterFoldConfig(
                    train_mode="rolling",
                    train_span=pd.Timedelta("180D"),
                    valid_span=pd.Timedelta("14D"),
                    test_span=pd.Timedelta("30D"),
                    step_span=pd.Timedelta("60D"),
                ),
                inner=InnerFoldConfig(n_splits=2, use_purging=True, use_embargo=True),
                sampling=SamplingPolicy(
                    mode="subsample",
                    event_fraction=0.5,
                    symbol_fraction=0.5,
                    event_sampling_method="time_block",
                ),
                symbol_policy=SymbolPolicy(
                    mode="subset_symbols",
                    subset_fraction=0.5,
                    min_symbols_per_split=2,
                ),
                purge_policy=PurgePolicy(
                    embargo_timedelta=pd.Timedelta("1D"),
                    purge_scope="global",
                ),
            ),
            timezone=timezone,
            strict_validation=False,
            min_rows_per_fold=25,
            min_symbols_per_fold=2,
        )


@dataclass
class OverlapHelper:
    starts_ns: np.ndarray
    ends_ns: np.ndarray


@dataclass
class PlannerState:
    events: pd.DataFrame
    schema: EventSchema
    t0_ns: np.ndarray
    t1_ns: np.ndarray
    symbol_codes: np.ndarray
    symbol_labels: np.ndarray
    group_codes: dict[str, np.ndarray]
    group_labels: dict[str, np.ndarray]
    overlap: OverlapHelper


@dataclass
class ResolvedFoldPartition:
    outer_fold_id: int
    symbol_mode: str
    train_symbol_codes: np.ndarray
    valid_symbol_codes: np.ndarray
    test_symbol_codes: np.ndarray
    holdout_groups: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class OuterFold:
    outer_fold_id: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    train_symbols: tuple[str, ...]
    valid_symbols: tuple[str, ...]
    test_symbols: tuple[str, ...]
    partition: ResolvedFoldPartition
    skipped: bool = False
    skip_reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InnerFold:
    outer_fold_id: int
    inner_fold_id: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    skipped: bool = False
    skip_reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsumerSlicePlan:
    consumer_role: str
    outer_fold_id: int
    inner_fold_id: Optional[int]
    fit_idx: np.ndarray
    predict_idx: np.ndarray
    oof_target_idx: np.ndarray
    tag: str
    symbols_fit: tuple[str, ...]
    symbols_predict: tuple[str, ...]
    metadata: dict[str, Any]
    skipped: bool = False
    skip_reason: Optional[str] = None


def _default_tprint(config: SlicePlannerConfig) -> Callable[[str], None]:
    if config.tprint is not None:
        return config.tprint
    if config.silent:
        return lambda _msg: None
    return print


def _context_seed(base_seed: int, fold_context: dict[str, Any]) -> int:
    raw = f"{base_seed}|{sorted(fold_context.items())}".encode("utf-8")
    return zlib.crc32(raw) & 0xFFFFFFFF


def optimise_event_dtypes(
    events: pd.DataFrame, config: SlicePlannerConfig
) -> pd.DataFrame:
    """Safely downcast numeric columns and optionally cast symbol/group to category."""
    out = events.copy()
    excluded = set(config.downcast_exclude_cols)

    if config.int_downcast:
        for col in out.select_dtypes(include=["int64", "int32", "int16"]).columns:
            if col not in excluded:
                out[col] = pd.to_numeric(out[col], downcast="integer")

    if config.float_downcast:
        for col in out.select_dtypes(include=["float64", "float32"]).columns:
            if col not in excluded:
                out[col] = pd.to_numeric(out[col], downcast="float")

    if config.cast_symbol_to_category and config.schema.symbol_col in out.columns:
        out[config.schema.symbol_col] = out[config.schema.symbol_col].astype("category")

    if config.cast_group_cols_to_category:
        for col in config.schema.group_cols:
            if col in out.columns:
                out[col] = out[col].astype("category")

    return out


def _split_symbol_base_quote(symbol: Any) -> tuple[str, Optional[str]]:
    sym = str(symbol or "").upper().replace("_", "/").strip()
    if "/" not in sym:
        return sym, None
    base, quote = sym.split("/", 1)
    return base, quote


def _deduplicate_alike_symbols(
    events: pd.DataFrame,
    schema: EventSchema,
    log: Callable[[str], None],
) -> pd.DataFrame:
    symbol_col = schema.symbol_col
    work = events.copy()
    split = work[symbol_col].map(_split_symbol_base_quote)
    work["_dedup_base"] = split.map(lambda x: x[0])
    work["_dedup_quote"] = split.map(lambda x: x[1])

    eligible = work["_dedup_quote"].isin(DEDUP_QUOTES)
    if not eligible.any():
        return work.drop(columns=["_dedup_base", "_dedup_quote"])

    keep_symbols: set[str] = set()
    removed_symbols: list[str] = []

    eligible_view = work.loc[
        eligible, [symbol_col, "_dedup_base", "_dedup_quote"]
    ].copy()
    counts = (
        eligible_view.groupby(
            [symbol_col, "_dedup_base", "_dedup_quote"], observed=False
        )
        .size()
        .reset_index(name="n_rows")
    )

    for base, grp in counts.groupby("_dedup_base", sort=False):
        if grp.shape[0] <= 1:
            keep_symbols.add(str(grp.iloc[0][symbol_col]))
            continue
        ranked = grp.assign(
            _quote_rank=grp["_dedup_quote"].map(
                lambda q: DEDUP_QUOTE_PRIORITY.get(str(q), 999)
            )
        ).sort_values(
            ["n_rows", "_quote_rank", symbol_col],
            ascending=[False, True, True],
            kind="mergesort",
        )
        winner = str(ranked.iloc[0][symbol_col])
        keep_symbols.add(winner)
        removed_symbols.extend(
            [str(sym) for sym in ranked.iloc[1:][symbol_col].tolist()]
        )

    drop_mask = eligible & ~work[symbol_col].astype(str).isin(keep_symbols)
    if drop_mask.any():
        before_rows = int(work.shape[0])
        before_symbols = int(work[symbol_col].nunique())
        work = work.loc[~drop_mask].copy()
        after_rows = int(work.shape[0])
        after_symbols = int(work[symbol_col].nunique())
        sample_removed = sorted(set(removed_symbols))[:10]
        log(
            "[slice_planner] deduplicated alike symbols "
            f"rows={before_rows}->{after_rows} symbols={before_symbols}->{after_symbols} "
            f"removed={len(set(removed_symbols))} sample_removed={sample_removed}"
        )

    return work.drop(columns=["_dedup_base", "_dedup_quote"])


def validate_events(
    events: pd.DataFrame,
    schema: EventSchema,
    config: SlicePlannerConfig,
    tprint: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """Validate required fields and build deterministic sorted event table."""
    log = tprint or _default_tprint(config)
    required = [schema.event_id_col, schema.symbol_col, schema.t0_col, schema.t1_col]
    missing = [col for col in required if col not in events.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = events.copy()
    out = _deduplicate_alike_symbols(out, schema, log)
    for col in required:
        if out[col].isna().any():
            raise ValueError(f"Null values detected in required column '{col}'")

    if not out[schema.event_id_col].is_unique:
        dups = out.loc[out[schema.event_id_col].duplicated(), schema.event_id_col].head(
            5
        )
        raise ValueError(f"Duplicate event_id values detected. Sample={dups.tolist()}")

    out[schema.t0_col] = pd.to_datetime(out[schema.t0_col], utc=True, errors="raise")
    out[schema.t1_col] = pd.to_datetime(out[schema.t1_col], utc=True, errors="raise")

    if config.timezone:
        out[schema.t0_col] = out[schema.t0_col].dt.tz_convert(config.timezone)
        out[schema.t1_col] = out[schema.t1_col].dt.tz_convert(config.timezone)

    bad_interval = out[schema.t1_col] < out[schema.t0_col]
    if bad_interval.any():
        sample = out.loc[
            bad_interval, [schema.event_id_col, schema.t0_col, schema.t1_col]
        ].head(3)
        raise ValueError(f"Found events with t1 < t0. Sample:\n{sample}")

    out = out.sort_values(
        by=[schema.t0_col, schema.symbol_col, schema.event_id_col],
        kind="mergesort",
    ).reset_index(drop=True)

    if config.dtype_optimisation:
        out = optimise_event_dtypes(out, config)

    out["_row_id"] = np.arange(out.shape[0], dtype=np.int64)
    out["_t0_ns"] = out[schema.t0_col].astype("int64")
    out["_t1_ns"] = out[schema.t1_col].astype("int64")

    sym = out[schema.symbol_col]
    if not isinstance(sym.dtype, CategoricalDtype):
        sym = sym.astype("category")
        out[schema.symbol_col] = sym
    out["_symbol_code"] = sym.cat.codes.astype(np.int32)

    for col in schema.group_cols:
        if col in out.columns:
            grp = out[col]
            if not isinstance(grp.dtype, CategoricalDtype):
                grp = grp.astype("category")
                out[col] = grp
            out[f"_group_code__{col}"] = grp.cat.codes.astype(np.int32)

    log(
        f"[slice_planner] validated events n={len(out)} symbols={out[schema.symbol_col].nunique()} "
        f"period={out[schema.t0_col].min()} -> {out[schema.t0_col].max()}"
    )
    return out


def build_overlap_helper(events: pd.DataFrame, schema: EventSchema) -> OverlapHelper:
    """Build start/end arrays for efficient overlap queries."""
    return OverlapHelper(
        starts_ns=events["_t0_ns"].to_numpy(dtype=np.int64, copy=True),
        ends_ns=events["_t1_ns"].to_numpy(dtype=np.int64, copy=True),
    )


def build_planner_state(
    events: pd.DataFrame, config: SlicePlannerConfig
) -> PlannerState:
    schema = config.schema
    symbol_cat = events[schema.symbol_col].astype("category")
    symbol_labels = symbol_cat.cat.categories.astype(str).to_numpy()

    group_codes: dict[str, np.ndarray] = {}
    group_labels: dict[str, np.ndarray] = {}
    for col in schema.group_cols:
        if col in events.columns:
            c = events[col].astype("category")
            group_codes[col] = events[f"_group_code__{col}"].to_numpy(
                dtype=np.int32, copy=False
            )
            group_labels[col] = c.cat.categories.astype(str).to_numpy()

    return PlannerState(
        events=events,
        schema=schema,
        t0_ns=events["_t0_ns"].to_numpy(dtype=np.int64, copy=False),
        t1_ns=events["_t1_ns"].to_numpy(dtype=np.int64, copy=False),
        symbol_codes=events["_symbol_code"].to_numpy(dtype=np.int32, copy=False),
        symbol_labels=symbol_labels,
        group_codes=group_codes,
        group_labels=group_labels,
        overlap=build_overlap_helper(events, schema),
    )


def _merge_intervals(
    starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if starts.size == 0:
        return starts, ends
    order = np.argsort(starts, kind="mergesort")
    s = starts[order]
    e = ends[order]
    out_s = [int(s[0])]
    out_e = [int(e[0])]
    for cur_s, cur_e in zip(s[1:], e[1:]):
        if cur_s <= out_e[-1]:
            if cur_e > out_e[-1]:
                out_e[-1] = int(cur_e)
        else:
            out_s.append(int(cur_s))
            out_e.append(int(cur_e))
    return np.asarray(out_s, dtype=np.int64), np.asarray(out_e, dtype=np.int64)


def _effective_embargo_ns(p: PurgePolicy) -> int:
    values = [int(p.embargo_timedelta.value)]
    if p.max_lookback_timedelta is not None:
        values.append(int(p.max_lookback_timedelta.value))
    if p.max_holding_timedelta is not None:
        values.append(int(p.max_holding_timedelta.value))
    return max(values)


def _ns_to_timestamp(ns: int, timezone: Optional[str]) -> pd.Timestamp:
    ts = pd.Timestamp(ns, tz="UTC")
    if timezone:
        return ts.tz_convert(timezone)
    return ts.tz_localize(None)


def _resolve_timezone(config: SlicePlannerConfig, state: PlannerState) -> Optional[str]:
    if config.timezone:
        return config.timezone
    tz = state.events[state.schema.t0_col].dt.tz
    return str(tz) if tz is not None else None


def _split_sampling_flags(split: str, sampling: SamplingPolicy) -> tuple[bool, bool]:
    if split == "train":
        return sampling.sample_train, sampling.sample_symbols_on_train
    if split == "inner_train":
        return sampling.sample_inner_train, sampling.sample_symbols_on_inner_train
    if split == "valid":
        return sampling.sample_valid, sampling.sample_symbols_on_valid
    if split == "inner_valid":
        return sampling.sample_inner_valid, sampling.sample_symbols_on_inner_valid
    if split == "test":
        return sampling.sample_test, sampling.sample_symbols_on_test
    return False, False


def _subsample_event_positions(n: int, target: int, method: str) -> np.ndarray:
    if target >= n:
        return np.arange(n, dtype=np.int64)
    if method == "head":
        return np.arange(target, dtype=np.int64)
    if method == "tail":
        return np.arange(n - target, n, dtype=np.int64)
    if method == "every_kth":
        k = max(1, int(np.floor(n / target)))
        idx = np.arange(0, n, k, dtype=np.int64)
        return (
            idx[:target]
            if idx.size >= target
            else np.linspace(0, n - 1, target, dtype=np.int64)
        )
    return np.arange(n, dtype=np.int64)


def _window_idx(t0_ns: np.ndarray, start_ns: int, end_ns: int) -> np.ndarray:
    return np.flatnonzero((t0_ns >= start_ns) & (t0_ns < end_ns)).astype(np.int64)


def _period_from_idx(
    state: PlannerState, idx: np.ndarray, timezone: Optional[str]
) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return None, None
    return _ns_to_timestamp(int(state.t0_ns[idx].min()), timezone), _ns_to_timestamp(
        int(state.t0_ns[idx].max()), timezone
    )


def _subsample_time_block_idx(
    state: PlannerState, idx: np.ndarray, target: int
) -> np.ndarray:
    """Deterministic trailing time-block subsample based on actual t0 values."""
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size <= target:
        return idx
    t0 = state.t0_ns[idx]
    uniq = np.unique(t0)
    # Find smallest trailing timestamp suffix reaching at least target rows.
    counts = np.array([(t0 == ts).sum() for ts in uniq], dtype=np.int64)
    csum_rev = np.cumsum(counts[::-1])
    k = int(np.searchsorted(csum_rev, target, side="left"))
    cutoff = uniq[max(0, uniq.size - 1 - k)]
    keep = idx[t0 >= cutoff]
    if keep.size > target:
        # deterministic trim from the front while preserving chronological order
        keep = keep[-target:]
    return np.sort(keep)


def purge_train_indices(
    state: PlannerState,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    purge_policy: PurgePolicy,
    train_symbol_codes: Optional[np.ndarray] = None,
    test_symbol_codes: Optional[np.ndarray] = None,
    train_group_codes: Optional[np.ndarray] = None,
    test_group_codes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Purge overlap + embargo from train rows against test rows under configured scope."""
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    if train_idx.size == 0 or test_idx.size == 0:
        return np.unique(train_idx)

    train_starts = state.overlap.starts_ns[train_idx]
    train_ends = state.overlap.ends_ns[train_idx]
    keep = np.ones(train_idx.size, dtype=bool)

    if purge_policy.purge_on_overlap:
        if purge_policy.purge_scope == "global":
            ms, me = _merge_intervals(
                state.overlap.starts_ns[test_idx], state.overlap.ends_ns[test_idx]
            )
            pos = np.searchsorted(ms, train_ends, side="right") - 1
            overlap = pos >= 0
            overlap[overlap] = me[pos[overlap]] >= train_starts[overlap]
            keep &= ~overlap
        else:
            scope_map_train: np.ndarray
            scope_map_test: np.ndarray
            if purge_policy.purge_scope == "same_symbol":
                if train_symbol_codes is None or test_symbol_codes is None:
                    train_symbol_codes = state.symbol_codes[train_idx]
                    test_symbol_codes = state.symbol_codes[test_idx]
                scope_map_train = train_symbol_codes
                scope_map_test = test_symbol_codes
            elif purge_policy.purge_scope == "same_group":
                if train_group_codes is None or test_group_codes is None:
                    if not purge_policy.purge_group_col:
                        raise ValueError(
                            "purge_scope='same_group' requires purge_group_col"
                        )
                    if purge_policy.purge_group_col not in state.group_codes:
                        raise ValueError(
                            f"purge_group_col '{purge_policy.purge_group_col}' not available"
                        )
                    code_arr = state.group_codes[purge_policy.purge_group_col]
                    train_group_codes = code_arr[train_idx]
                    test_group_codes = code_arr[test_idx]
                scope_map_train = train_group_codes
                scope_map_test = test_group_codes
            else:
                raise ValueError(f"Unsupported purge scope: {purge_policy.purge_scope}")

            for key in np.intersect1d(
                np.unique(scope_map_train), np.unique(scope_map_test)
            ):
                tr_loc = np.flatnonzero(scope_map_train == key)
                te_loc = np.flatnonzero(scope_map_test == key)
                if tr_loc.size == 0 or te_loc.size == 0:
                    continue
                ms, me = _merge_intervals(
                    state.overlap.starts_ns[test_idx[te_loc]],
                    state.overlap.ends_ns[test_idx[te_loc]],
                )
                tr_starts = train_starts[tr_loc]
                tr_ends = train_ends[tr_loc]
                pos = np.searchsorted(ms, tr_ends, side="right") - 1
                overlap = pos >= 0
                overlap[overlap] = me[pos[overlap]] >= tr_starts[overlap]
                keep[tr_loc] &= ~overlap

    embargo_ns = _effective_embargo_ns(purge_policy)
    if embargo_ns > 0:
        anchor = (
            int(state.overlap.ends_ns[test_idx].max())
            if purge_policy.purge_with_test_interval_end
            else int(state.overlap.starts_ns[test_idx].max())
        )
        embargo_end = anchor + embargo_ns
        embargo_mask = (train_starts > anchor) & (train_starts <= embargo_end)
        keep &= ~embargo_mask

    return np.unique(train_idx[keep])


def _codes_to_symbols(codes: np.ndarray, labels: np.ndarray) -> tuple[str, ...]:
    if codes.size == 0:
        return ()
    codes = codes[codes >= 0]
    if codes.size == 0:
        return ()
    return tuple(labels[np.unique(codes)].tolist())


def _rotating_symbol_subset(codes: np.ndarray, n_pick: int, offset: int) -> np.ndarray:
    """Select a deterministic, cyclic subset from sorted codes."""
    if codes.size == 0 or n_pick <= 0:
        return np.array([], dtype=np.int32)
    n_total = int(codes.size)
    n_pick = min(max(1, int(n_pick)), n_total)
    codes = np.sort(np.asarray(codes, dtype=np.int32))
    if n_pick >= n_total:
        return np.array(codes, copy=False)
    start = offset % n_total
    sel = (np.arange(n_pick, dtype=np.int64) + start) % n_total
    return np.sort(codes[sel])


def _resolve_outer_partition(
    state: PlannerState,
    policy: SymbolPolicy,
    outer_fold_id: int,
    train_ref_idx: np.ndarray,
    valid_ref_idx: np.ndarray,
    test_ref_idx: np.ndarray,
) -> ResolvedFoldPartition:
    """Resolve fold-stable symbol/group partition once per outer fold."""
    split_ref = np.unique(np.concatenate([train_ref_idx, valid_ref_idx, test_ref_idx]))
    universe_codes = np.unique(state.symbol_codes[split_ref])
    rng = np.random.default_rng(
        _context_seed(policy.random_state, {"outer_fold": outer_fold_id})
    )

    if policy.mode == "all_symbols":
        train_codes = universe_codes
        valid_codes = universe_codes
        test_codes = universe_codes
        holdout_groups: dict[str, np.ndarray] = {}
    elif policy.mode == "subset_symbols":
        frac = 1.0 if policy.subset_fraction is None else policy.subset_fraction
        n_pick = int(np.ceil(frac * universe_codes.size))
        # Ensure we don't try to pick more than available (handles edge cases where
        # min_symbols > total) and guarantee each fold starts from a different
        # subset offset so symbols are spread across folds.
        n_pick = max(policy.min_symbols_per_split, min(n_pick, universe_codes.size))
        n_pick = min(n_pick, universe_codes.size)
        subset = _rotating_symbol_subset(universe_codes, n_pick, outer_fold_id)

        train_codes = subset
        valid_codes = subset
        test_codes = subset
        holdout_groups = {}
    elif policy.mode == "disjoint_symbol_holdout":
        shuffled = universe_codes.copy()
        rng.shuffle(shuffled)
        cut = max(policy.min_symbols_per_split, shuffled.size // 2)
        cut = min(cut, max(1, shuffled.size - policy.min_symbols_per_split))
        train_codes = np.sort(shuffled[:cut]).astype(np.int32)
        test_codes = np.sort(shuffled[cut:]).astype(np.int32)
        if test_codes.size == 0:
            test_codes = train_codes
        valid_codes = test_codes if policy.unseen_test_symbols else train_codes
        holdout_groups = {}
    elif policy.mode == "group_holdout":
        if not policy.group_col or policy.group_col not in state.group_codes:
            raise ValueError(
                "group_holdout requires a valid group_col present in events"
            )
        # Fold-local resolution is required to avoid leakage/inconsistency across windows.
        fold_groups = state.group_codes[policy.group_col][split_ref]
        fold_symbols = state.symbol_codes[split_ref]
        uniq = np.unique(fold_groups[fold_groups >= 0])
        if uniq.size == 0:
            raise ValueError("group_holdout has no valid groups in fold-local universe")
        held = np.asarray([rng.choice(uniq)], dtype=np.int32)
        holdout_groups = {policy.group_col: held}
        in_test = np.isin(fold_groups, held)
        test_codes = np.unique(fold_symbols[in_test])
        train_codes = np.unique(fold_symbols[~in_test])
        valid_codes = test_codes if policy.unseen_test_symbols else train_codes
    elif policy.mode == "fixed_symbol_partition":
        label_to_code = {lab: i for i, lab in enumerate(state.symbol_labels)}

        def to_codes(
            values: Optional[tuple[str, ...]], fallback: np.ndarray
        ) -> np.ndarray:
            if not values:
                return fallback
            out = np.array(
                [label_to_code[v] for v in values if v in label_to_code], dtype=np.int32
            )
            return np.unique(out) if out.size > 0 else fallback

        train_codes = to_codes(policy.train_symbols, universe_codes)
        valid_codes = to_codes(policy.valid_symbols, train_codes)
        test_codes = to_codes(policy.test_symbols, universe_codes)
        holdout_groups = {}
    else:
        raise ValueError(f"Unsupported symbol mode: {policy.mode}")

    return ResolvedFoldPartition(
        outer_fold_id=outer_fold_id,
        symbol_mode=policy.mode,
        train_symbol_codes=np.unique(train_codes).astype(np.int32),
        valid_symbol_codes=np.unique(valid_codes).astype(np.int32),
        test_symbol_codes=np.unique(test_codes).astype(np.int32),
        holdout_groups=holdout_groups,
    )


def apply_symbol_policy(
    state: PlannerState,
    idx: np.ndarray,
    partition: ResolvedFoldPartition,
    split_role: Literal["train", "valid", "test", "inner_train", "inner_valid"],
    symbol_policy: SymbolPolicy,
) -> np.ndarray:
    """Apply resolved fold-level symbol/group policy deterministically."""
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return idx

    if split_role in {"train", "inner_train"}:
        allowed = partition.train_symbol_codes
    elif split_role in {"valid", "inner_valid"}:
        allowed = partition.valid_symbol_codes
    else:
        allowed = partition.test_symbol_codes

    out = idx[np.isin(state.symbol_codes[idx], allowed)]

    if symbol_policy.mode == "group_holdout" and symbol_policy.group_col:
        held = partition.holdout_groups.get(
            symbol_policy.group_col, np.array([], dtype=np.int32)
        )
        g = state.group_codes[symbol_policy.group_col]
        if split_role in {"test", "valid", "inner_valid"}:
            out = out[np.isin(g[out], held)]
        else:
            out = out[~np.isin(g[out], held)]

    uniq = np.unique(state.symbol_codes[out])
    if out.size > 0 and uniq.size < symbol_policy.min_symbols_per_split:
        return np.array([], dtype=np.int64)
    return np.sort(out)


def apply_sampling_policy(
    state: PlannerState,
    idx: np.ndarray,
    sampling_policy: SamplingPolicy,
    symbol_policy: Optional[SymbolPolicy],
    fold_context: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply deterministic split-aware subsampling while preserving chronology."""
    idx = np.asarray(idx, dtype=np.int64)
    split = str(fold_context.get("split", "train"))
    meta = {
        "split": split,
        "event_sampling_applied": False,
        "event_sampling_method": "none",
        "symbol_sampling_applied": False,
    }

    if idx.size == 0 or sampling_policy.mode == "full":
        return idx, meta

    sample_events, sample_symbols = _split_sampling_flags(split, sampling_policy)
    out = idx.copy()

    if sample_symbols and (
        sampling_policy.symbol_fraction < 1.0 or sampling_policy.max_symbols is not None
    ):
        split = str(fold_context.get("split", ""))
        sym = np.sort(np.unique(state.symbol_codes[out]))
        n_pick = int(np.ceil(sym.size * sampling_policy.symbol_fraction))
        if sampling_policy.max_symbols is not None:
            n_pick = min(n_pick, sampling_policy.max_symbols)
        min_pick = 1
        if symbol_policy is not None and split in {"train", "inner_train"}:
            min_pick = max(1, int(symbol_policy.min_symbols_per_split))
        n_pick = max(min_pick, min(n_pick, sym.size))
        if n_pick < sym.size:
            outer_fold = int(fold_context.get("outer_fold", 0))
            inner_fold = int(fold_context.get("inner_fold", 0))
            split_offset = {
                "train": 0,
                "inner_train": 1,
                "valid": 2,
                "inner_valid": 3,
                "test": 4,
            }.get(split, 5)
            offset = outer_fold * 97 + inner_fold * 127 + split_offset
            keep = _rotating_symbol_subset(sym, n_pick, offset)
        else:
            keep = sym
        out = out[np.isin(state.symbol_codes[out], keep)]
        # Keep deterministic ordering after symbol filtering.
        meta["symbol_sampling_applied"] = True

    if out.size == 0:
        return out, meta

    if sample_events and (
        sampling_policy.event_fraction < 1.0 or sampling_policy.max_events is not None
    ):
        target = int(np.ceil(out.size * sampling_policy.event_fraction))
        if sampling_policy.max_events is not None:
            target = min(target, sampling_policy.max_events)
        target = max(1, min(target, out.size))
        method = sampling_policy.event_sampling_method
        if method == "none":
            method = "time_block"
        if method == "time_block":
            out = _subsample_time_block_idx(state, out, target)
        else:
            pos = _subsample_event_positions(out.size, target, method)
            out = out[np.sort(pos)]
        meta["event_sampling_applied"] = True
        meta["event_sampling_method"] = method

    return np.sort(out), meta


def _fold_periods(
    outer_cfg: OuterFoldConfig, min_ns: int, max_ns: int
) -> list[tuple[int, int, int, int, int, int]]:
    periods: list[tuple[int, int, int, int, int, int]] = []
    train_span = int(outer_cfg.train_span.value)
    valid_span = (
        int(outer_cfg.valid_span.value) if outer_cfg.valid_span is not None else 0
    )
    test_span = int(outer_cfg.test_span.value)
    step_span = int(outer_cfg.step_span.value)

    test_start = min_ns + train_span + valid_span
    fid = 0
    while test_start <= max_ns:
        if outer_cfg.n_folds is not None and fid >= outer_cfg.n_folds:
            break
        test_end = test_start + test_span
        valid_start = test_start - valid_span
        valid_end = test_start
        train_end = valid_start
        train_start = (
            max(min_ns, train_end - train_span)
            if outer_cfg.train_mode == "rolling"
            else min_ns
        )
        if train_start >= train_end:
            break
        periods.append((fid, train_start, train_end, valid_start, valid_end, test_end))
        test_start += step_span
        fid += 1
    return periods


def _emit_fold_log(
    log: Callable[[str], None], of: OuterFold, preset: str, sampling_mode: str
) -> None:
    meta = of.metadata
    log(
        "[slice_planner] "
        f"outer={of.outer_fold_id} preset={preset} mode={of.partition.symbol_mode}/{sampling_mode} "
        f"valid_role={meta.get('valid_role')} tz={meta['train_window_start'].tz} "
        f"train={of.train_idx.size} valid={of.valid_idx.size} test={of.test_idx.size} "
        f"sym(train/valid/test)={len(of.train_symbols)}/{len(of.valid_symbols)}/{len(of.test_symbols)} "
        f"window train[{meta['train_window_start']}->{meta['train_window_end']}] "
        f"valid[{meta.get('valid_window_start')}->{meta.get('valid_window_end')}] "
        f"test[{meta['test_window_start']}->{meta['test_window_end']}] "
        f"actual train[{meta['train_actual_start']}->{meta['train_actual_end']}] "
        f"valid[{meta['valid_actual_start']}->{meta['valid_actual_end']}] "
        f"test[{meta['test_actual_start']}->{meta['test_actual_end']}] "
        f"sampled_train={meta['train_sampling']['event_sampling_applied'] or meta['train_sampling']['symbol_sampling_applied']} "
        f"valid_untouched={not meta['valid_sampling']['event_sampling_applied'] and not meta['valid_sampling']['symbol_sampling_applied']} "
        f"test_untouched={not meta['test_sampling']['event_sampling_applied'] and not meta['test_sampling']['symbol_sampling_applied']} "
        f"skipped={of.skipped} reason={of.skip_reason}"
    )


def make_outer_walkforward_folds(
    state: PlannerState,
    config: SlicePlannerConfig,
    tprint: Optional[Callable[[str], None]] = None,
) -> list[OuterFold]:
    """Create outer walk-forward folds with first-class train/valid/test windows."""
    log = tprint or _default_tprint(config)
    outer_cfg = config.preset.outer
    policy = config.preset.symbol_policy
    sampling = config.preset.sampling
    purge = config.preset.purge_policy

    periods = _fold_periods(outer_cfg, int(state.t0_ns.min()), int(state.t0_ns.max()))
    timezone = _resolve_timezone(config, state)
    folds: list[OuterFold] = []
    for fold_id, tr_s, tr_e, va_s, va_e, te_e in periods:
        te_s = va_e
        train_ref = _window_idx(state.t0_ns, tr_s, tr_e)
        valid_ref = (
            _window_idx(state.t0_ns, va_s, va_e)
            if outer_cfg.valid_span is not None
            else np.array([], dtype=np.int64)
        )
        test_ref = _window_idx(state.t0_ns, te_s, te_e)

        partition = _resolve_outer_partition(
            state, policy, fold_id, train_ref, valid_ref, test_ref
        )

        train_idx = apply_symbol_policy(state, train_ref, partition, "train", policy)
        valid_idx = apply_symbol_policy(state, valid_ref, partition, "valid", policy)
        test_idx = apply_symbol_policy(state, test_ref, partition, "test", policy)

        train_idx, train_sampling_meta = apply_sampling_policy(
            state,
            train_idx,
            sampling,
            policy,
            {"outer_fold": fold_id, "split": "train"},
        )
        valid_idx, valid_sampling_meta = apply_sampling_policy(
            state,
            valid_idx,
            sampling,
            None,
            {"outer_fold": fold_id, "split": "valid"},
        )
        test_idx, test_sampling_meta = apply_sampling_policy(
            state,
            test_idx,
            sampling,
            None,
            {"outer_fold": fold_id, "split": "test"},
        )

        train_idx = purge_train_indices(
            state=state,
            train_idx=train_idx,
            test_idx=(
                np.unique(np.concatenate([valid_idx, test_idx]))
                if valid_idx.size > 0
                else test_idx
            ),
            purge_policy=purge,
            train_symbol_codes=state.symbol_codes[train_idx],
            test_symbol_codes=state.symbol_codes[
                (
                    np.unique(np.concatenate([valid_idx, test_idx]))
                    if valid_idx.size > 0
                    else test_idx
                )
            ],
            train_group_codes=(
                state.group_codes[purge.purge_group_col][train_idx]
                if purge.purge_scope == "same_group"
                and purge.purge_group_col in state.group_codes
                else None
            ),
            test_group_codes=(
                state.group_codes[purge.purge_group_col][
                    (
                        np.unique(np.concatenate([valid_idx, test_idx]))
                        if valid_idx.size > 0
                        else test_idx
                    )
                ]
                if purge.purge_scope == "same_group"
                and purge.purge_group_col in state.group_codes
                else None
            ),
        )

        train_symbols = _codes_to_symbols(
            state.symbol_codes[train_idx], state.symbol_labels
        )
        valid_symbols = _codes_to_symbols(
            state.symbol_codes[valid_idx], state.symbol_labels
        )
        test_symbols = _codes_to_symbols(
            state.symbol_codes[test_idx], state.symbol_labels
        )

        skipped = False
        reason = None
        if train_idx.size < config.min_rows_per_fold:
            skipped, reason = True, "too_few_train_rows"
        elif test_idx.size < config.min_rows_per_fold:
            skipped, reason = True, "too_few_test_rows"
        elif len(train_symbols) < config.min_symbols_per_fold:
            skipped, reason = True, "too_few_train_symbols"
        elif len(test_symbols) == 0:
            skipped, reason = True, "no_test_symbols"

        max_tail_days = 0.0
        if test_idx.size > 0:
            test_end = int(state.t0_ns[test_idx].max())
            long_tail = state.t1_ns[test_idx] - test_end
            max_tail_days = float(np.max(long_tail) / pd.Timedelta("1D").value)

        of = OuterFold(
            outer_fold_id=fold_id,
            train_idx=train_idx,
            valid_idx=valid_idx,
            test_idx=test_idx,
            train_symbols=train_symbols,
            valid_symbols=valid_symbols,
            test_symbols=test_symbols,
            partition=partition,
            skipped=skipped,
            skip_reason=reason,
            metadata={
                "train_window_start": _ns_to_timestamp(tr_s, timezone),
                "train_window_end": _ns_to_timestamp(tr_e, timezone),
                "valid_window_start": (
                    _ns_to_timestamp(va_s, timezone)
                    if outer_cfg.valid_span is not None
                    else None
                ),
                "valid_window_end": (
                    _ns_to_timestamp(va_e, timezone)
                    if outer_cfg.valid_span is not None
                    else None
                ),
                "test_window_start": _ns_to_timestamp(te_s, timezone),
                "test_window_end": _ns_to_timestamp(te_e, timezone),
                "train_actual_start": _period_from_idx(state, train_idx, timezone)[0],
                "train_actual_end": _period_from_idx(state, train_idx, timezone)[1],
                "valid_actual_start": _period_from_idx(state, valid_idx, timezone)[0],
                "valid_actual_end": _period_from_idx(state, valid_idx, timezone)[1],
                "test_actual_start": _period_from_idx(state, test_idx, timezone)[0],
                "test_actual_end": _period_from_idx(state, test_idx, timezone)[1],
                "n_train": int(train_idx.size),
                "n_valid": int(valid_idx.size),
                "n_test": int(test_idx.size),
                "n_symbols_train": len(train_symbols),
                "n_symbols_valid": len(valid_symbols),
                "n_symbols_test": len(test_symbols),
                "max_test_t1_tail_days": max_tail_days,
                "valid_role": (
                    "outer_valid_tail" if outer_cfg.valid_span is not None else None
                ),
                "train_sampling": train_sampling_meta,
                "valid_sampling": valid_sampling_meta,
                "test_sampling": test_sampling_meta,
            },
        )
        _emit_fold_log(log, of, config.preset.preset_name, sampling.mode)
        folds.append(of)

    return folds


def make_inner_purged_folds(
    state: PlannerState,
    outer_fold: OuterFold,
    config: SlicePlannerConfig,
    tprint: Optional[Callable[[str], None]] = None,
) -> list[InnerFold]:
    """Create chronological inner folds inside outer-train with purge/embargo."""
    log = tprint or _default_tprint(config)
    inner = config.preset.inner
    policy = config.preset.symbol_policy
    sampling = config.preset.sampling
    purge = config.preset.purge_policy

    outer_train = np.asarray(outer_fold.train_idx, dtype=np.int64)
    if outer_train.size == 0:
        return []

    uniq_t0 = np.unique(state.t0_ns[outer_train])
    if uniq_t0.size < inner.n_splits + 1:
        return []

    splits = np.array_split(uniq_t0, inner.n_splits + 1)
    timezone = _resolve_timezone(config, state)
    out: list[InnerFold] = []
    for inner_id in range(1, inner.n_splits + 1):
        valid_times = splits[inner_id]
        if valid_times.size == 0:
            continue

        v_start = int(valid_times.min())
        v_end = int(valid_times.max()) + 1
        tr = outer_train[state.t0_ns[outer_train] < v_start]
        va = outer_train[
            (state.t0_ns[outer_train] >= v_start) & (state.t0_ns[outer_train] < v_end)
        ]

        tr = apply_symbol_policy(state, tr, outer_fold.partition, "inner_train", policy)
        va = apply_symbol_policy(state, va, outer_fold.partition, "inner_valid", policy)

        tr, tr_sampling_meta = apply_sampling_policy(
            state,
            tr,
            sampling,
            policy,
            {
                "outer_fold": outer_fold.outer_fold_id,
                "inner_fold": inner_id - 1,
                "split": "inner_train",
            },
        )
        va, va_sampling_meta = apply_sampling_policy(
            state,
            va,
            sampling,
            policy,
            {
                "outer_fold": outer_fold.outer_fold_id,
                "inner_fold": inner_id - 1,
                "split": "inner_valid",
            },
        )

        if inner.use_purging:
            pp = (
                purge
                if inner.use_embargo
                else replace(purge, embargo_timedelta=pd.Timedelta(0))
            )
            tr = purge_train_indices(
                state=state,
                train_idx=tr,
                test_idx=va,
                purge_policy=pp,
                train_symbol_codes=state.symbol_codes[tr],
                test_symbol_codes=state.symbol_codes[va],
                train_group_codes=(
                    state.group_codes[pp.purge_group_col][tr]
                    if pp.purge_scope == "same_group"
                    and pp.purge_group_col in state.group_codes
                    else None
                ),
                test_group_codes=(
                    state.group_codes[pp.purge_group_col][va]
                    if pp.purge_scope == "same_group"
                    and pp.purge_group_col in state.group_codes
                    else None
                ),
            )

        skipped = False
        reason = None
        if tr.size < config.min_rows_per_fold:
            skipped, reason = True, "too_few_inner_train_rows"
        elif va.size == 0:
            skipped, reason = True, "empty_inner_valid"

        inf = InnerFold(
            outer_fold_id=outer_fold.outer_fold_id,
            inner_fold_id=inner_id - 1,
            train_idx=np.sort(tr),
            valid_idx=np.sort(va),
            skipped=skipped,
            skip_reason=reason,
            metadata={
                "train_window_start": outer_fold.metadata.get("train_window_start"),
                "train_window_end": _ns_to_timestamp(v_start, timezone),
                "valid_window_start": _ns_to_timestamp(v_start, timezone),
                "valid_window_end": _ns_to_timestamp(v_end, timezone),
                "train_actual_start": _period_from_idx(state, tr, timezone)[0],
                "train_actual_end": _period_from_idx(state, tr, timezone)[1],
                "valid_actual_start": _period_from_idx(state, va, timezone)[0],
                "valid_actual_end": _period_from_idx(state, va, timezone)[1],
                "n_train": int(tr.size),
                "n_valid": int(va.size),
                "n_symbols_train": int(np.unique(state.symbol_codes[tr]).size),
                "n_symbols_valid": int(np.unique(state.symbol_codes[va]).size),
                "valid_role": "inner_oof_valid",
                "train_sampling": tr_sampling_meta,
                "valid_sampling": va_sampling_meta,
            },
        )
        log(
            "[slice_planner] "
            f"outer={outer_fold.outer_fold_id} inner={inf.inner_fold_id} "
            f"train={inf.metadata['n_train']} valid={inf.metadata['n_valid']} "
            f"symbols={inf.metadata['n_symbols_train']}/{inf.metadata['n_symbols_valid']} "
            f"valid_role={inf.metadata['valid_role']} tz={inf.metadata['valid_window_start'].tz if inf.metadata['valid_window_start'] is not None else None} "
            f"window_valid={inf.metadata['valid_window_start']}->{inf.metadata['valid_window_end']} actual_train={inf.metadata['train_actual_start']}->{inf.metadata['train_actual_end']} actual_valid={inf.metadata['valid_actual_start']}->{inf.metadata['valid_actual_end']} "
            f"sampled_train={inf.metadata['train_sampling']['event_sampling_applied'] or inf.metadata['train_sampling']['symbol_sampling_applied']} "
            f"valid_untouched={not inf.metadata['valid_sampling']['event_sampling_applied'] and not inf.metadata['valid_sampling']['symbol_sampling_applied']} "
            f"skipped={inf.skipped} reason={inf.skip_reason}"
        )
        out.append(inf)

    return out


def _idx_period(
    state: PlannerState, idx: np.ndarray, timezone: Optional[str]
) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    return _period_from_idx(state, idx, timezone)


def build_consumer_slice_plan(
    state: PlannerState,
    outer_folds: list[OuterFold],
    inner_by_outer: dict[int, list[InnerFold]],
    consumer_role: str,
    config: SlicePlannerConfig,
    tprint: Optional[Callable[[str], None]] = None,
) -> list[ConsumerSlicePlan]:
    """Build consumer-specific slice plans with explicit OOF eligibility metadata."""
    if consumer_role not in CONSUMER_ROLES:
        raise ValueError(f"Unsupported consumer role: {consumer_role}")

    log = tprint or _default_tprint(config)
    policy_mode = config.consumer_overrides.get("policy_tuning_mode", "inner_oof")
    timezone = _resolve_timezone(config, state)
    plans: list[ConsumerSlicePlan] = []

    if consumer_role == "full_inference_fit":
        lookback_years = float(
            config.consumer_overrides.get("full_inference_lookback_years", 4.0)
        )
        lookback_ns = int(pd.Timedelta(days=365.25 * lookback_years).value)
        max_t0 = int(np.max(state.t0_ns))
        cutoff = max_t0 - lookback_ns
        fit_idx = np.flatnonzero(state.t0_ns >= cutoff).astype(np.int64)
        fit_start, fit_end = _idx_period(state, fit_idx, timezone)
        fit_syms = _codes_to_symbols(state.symbol_codes[fit_idx], state.symbol_labels)
        plans.append(
            ConsumerSlicePlan(
                consumer_role=consumer_role,
                outer_fold_id=-1,
                inner_fold_id=None,
                fit_idx=fit_idx,
                predict_idx=np.array([], dtype=np.int64),
                oof_target_idx=np.array([], dtype=np.int64),
                tag="fit_full_inference",
                symbols_fit=fit_syms,
                symbols_predict=(),
                metadata={
                    "preset": config.preset.preset_name,
                    "plan_kind": "fit_only",
                    "fit_role": "full_inference_fit",
                    "predict_role": None,
                    "n_fit": int(fit_idx.size),
                    "n_predict": 0,
                    "n_symbols_fit": len(fit_syms),
                    "n_symbols_predict": 0,
                    "fit_window_start": _ns_to_timestamp(cutoff, timezone),
                    "fit_window_end": _ns_to_timestamp(max_t0, timezone),
                    "predict_window_start": None,
                    "predict_window_end": None,
                    "fit_actual_start": fit_start,
                    "fit_actual_end": fit_end,
                    "predict_actual_start": None,
                    "predict_actual_end": None,
                },
            )
        )
        log(
            f"[slice_planner] consumer=full_inference_fit outer=-1 inner=None tag=fit_full_inference "
            f"plan_kind=fit_only fit={fit_idx.size} pred=0 sym={len(fit_syms)}/0 "
            f"fit_window={_ns_to_timestamp(cutoff, timezone)}->{_ns_to_timestamp(max_t0, timezone)} "
            f"fit_actual={fit_start}->{fit_end} predict=empty"
        )
        return plans

    for of in outer_folds:
        if of.skipped:
            continue

        inners = [x for x in inner_by_outer.get(of.outer_fold_id, []) if not x.skipped]
        oof_covered = (
            np.unique(np.concatenate([x.valid_idx for x in inners]))
            if inners
            else np.array([], dtype=np.int64)
        )

        def add(
            tag: str,
            fit_idx: np.ndarray,
            pred_idx: np.ndarray,
            inner_id: Optional[int],
            fit_role: str,
            predict_role: Optional[str],
            fit_window_start: Optional[pd.Timestamp],
            fit_window_end: Optional[pd.Timestamp],
            predict_window_start: Optional[pd.Timestamp],
            predict_window_end: Optional[pd.Timestamp],
            fit_only: bool = False,
            extra: Optional[dict[str, Any]] = None,
        ) -> None:
            fit_idx = np.asarray(fit_idx, dtype=np.int64)
            if fit_only:
                pred_idx = np.array([], dtype=np.int64)
            else:
                pred_idx = np.asarray(pred_idx, dtype=np.int64)
            fit_syms = _codes_to_symbols(
                state.symbol_codes[fit_idx], state.symbol_labels
            )
            pred_syms = _codes_to_symbols(
                state.symbol_codes[pred_idx], state.symbol_labels
            )
            fit_start, fit_end = _idx_period(state, fit_idx, timezone)
            pred_start, pred_end = _idx_period(state, pred_idx, timezone)
            metadata = {
                "preset": config.preset.preset_name,
                "symbol_policy_mode": config.preset.symbol_policy.mode,
                "sampling_mode": config.preset.sampling.mode,
                "n_fit": int(fit_idx.size),
                "n_predict": int(pred_idx.size),
                "n_symbols_fit": len(fit_syms),
                "n_symbols_predict": len(pred_syms),
                "fit_start": fit_start,
                "fit_end": fit_end,
                "predict_start": pred_start,
                "predict_end": pred_end,
                "n_outer_train_all": int(of.train_idx.size),
                "n_outer_train_oof_eligible": int(oof_covered.size),
                "fit_role": fit_role,
                "predict_role": predict_role,
                "plan_kind": "fit_only" if fit_only else "fit_predict",
                "fit_window_start": fit_window_start,
                "fit_window_end": fit_window_end,
                "predict_window_start": predict_window_start,
                "predict_window_end": predict_window_end,
                "fit_actual_start": fit_start,
                "fit_actual_end": fit_end,
                "predict_actual_start": pred_start,
                "predict_actual_end": pred_end,
            }
            if extra:
                metadata.update(extra)
            plans.append(
                ConsumerSlicePlan(
                    consumer_role=consumer_role,
                    outer_fold_id=of.outer_fold_id,
                    inner_fold_id=inner_id,
                    fit_idx=fit_idx,
                    predict_idx=pred_idx,
                    oof_target_idx=(
                        np.array([], dtype=np.int64) if fit_only else pred_idx
                    ),
                    tag=tag,
                    symbols_fit=fit_syms,
                    symbols_predict=pred_syms,
                    metadata=metadata,
                )
            )
            log(
                "[slice_planner] "
                f"consumer={consumer_role} outer={of.outer_fold_id} inner={inner_id} tag={tag} "
                f"plan_kind={'fit_only' if fit_only else 'fit_predict'} "
                f"fit_role={fit_role} predict_role={predict_role} "
                f"fit={fit_idx.size} pred={pred_idx.size} sym={len(fit_syms)}/{len(pred_syms)} "
                f"fit_window={fit_window_start}->{fit_window_end} fit_actual={fit_start}->{fit_end} "
                f"pred_window={predict_window_start}->{predict_window_end} pred_actual={pred_start}->{pred_end} "
                f"outer_valid_tail={of.metadata.get('valid_role') == 'outer_valid_tail'} "
                f"symbol_mode={config.preset.symbol_policy.mode} sampling_mode={config.preset.sampling.mode} "
                f"skipped=False reason=None"
            )

        if consumer_role in {"regime_search", "barrier_search"}:
            for inf in inners:
                add(
                    "fit_inner",
                    inf.train_idx,
                    inf.valid_idx,
                    inf.inner_fold_id,
                    "inner_train",
                    "inner_oof_valid",
                    inf.metadata.get("train_window_start"),
                    inf.metadata.get("train_window_end"),
                    inf.metadata.get("valid_window_start"),
                    inf.metadata.get("valid_window_end"),
                )

        elif consumer_role == "base_model_fit":
            for inf in inners:
                add(
                    "fit_inner",
                    inf.train_idx,
                    inf.valid_idx,
                    inf.inner_fold_id,
                    "inner_train",
                    "inner_oof_valid",
                    inf.metadata.get("train_window_start"),
                    inf.metadata.get("train_window_end"),
                    inf.metadata.get("valid_window_start"),
                    inf.metadata.get("valid_window_end"),
                )
                add(
                    "predict_oof",
                    inf.train_idx,
                    inf.valid_idx,
                    inf.inner_fold_id,
                    "inner_train",
                    "inner_oof_valid",
                    inf.metadata.get("train_window_start"),
                    inf.metadata.get("train_window_end"),
                    inf.metadata.get("valid_window_start"),
                    inf.metadata.get("valid_window_end"),
                )
            add(
                "fit_full_outer",
                of.train_idx,
                of.train_idx,
                None,
                "full_outer_train",
                None,
                of.metadata.get("train_window_start"),
                of.metadata.get("train_window_end"),
                None,
                None,
                fit_only=True,
            )
            add(
                "predict_outer_test",
                of.train_idx,
                of.test_idx,
                None,
                "full_outer_train",
                "outer_test",
                of.metadata.get("train_window_start"),
                of.metadata.get("train_window_end"),
                of.metadata.get("test_window_start"),
                of.metadata.get("test_window_end"),
            )

        elif consumer_role == "meta_model_fit":
            add(
                "fit_outer_oof_only",
                oof_covered,
                oof_covered,
                None,
                "outer_train_oof_eligible",
                "outer_train_oof_eligible",
                of.metadata.get("train_window_start"),
                of.metadata.get("train_window_end"),
                of.metadata.get("train_window_start"),
                of.metadata.get("train_window_end"),
                extra={
                    "outer_train_all_idx": of.train_idx,
                    "outer_train_oof_eligible_idx": oof_covered,
                },
            )
            add(
                "predict_outer_test",
                of.train_idx,
                of.test_idx,
                None,
                "full_outer_train",
                "outer_test",
                of.metadata.get("train_window_start"),
                of.metadata.get("train_window_end"),
                of.metadata.get("test_window_start"),
                of.metadata.get("test_window_end"),
            )

        elif consumer_role == "ridge_sizer_fit":
            add(
                "fit_outer_oof_only",
                oof_covered,
                oof_covered,
                None,
                "outer_train_oof_eligible",
                "outer_train_oof_eligible",
                of.metadata.get("train_window_start"),
                of.metadata.get("train_window_end"),
                of.metadata.get("train_window_start"),
                of.metadata.get("train_window_end"),
                extra={
                    "outer_train_all_idx": of.train_idx,
                    "outer_train_oof_eligible_idx": oof_covered,
                },
            )
            n_test = len(of.test_idx)
            n_policy = max(0, int(n_test * config.policy_optimiser_holdout_frac))
            sizer_test_idx = (
                of.test_idx[: n_test - n_policy] if n_policy > 0 else of.test_idx
            )
            add(
                "predict_outer_test",
                of.train_idx,
                sizer_test_idx,
                None,
                "full_outer_train",
                "outer_test",
                of.metadata.get("train_window_start"),
                of.metadata.get("train_window_end"),
                of.metadata.get("test_window_start"),
                of.metadata.get("test_window_end"),
                extra={"held_out_for_policy": n_policy},
            )

        elif consumer_role == "utility_policy_tuning":
            if policy_mode == "outer_valid_tail" and of.valid_idx.size > 0:
                add(
                    "fit_outer_train_for_policy",
                    of.train_idx,
                    of.valid_idx,
                    None,
                    "outer_train",
                    "outer_valid_tail",
                    of.metadata.get("train_window_start"),
                    of.metadata.get("train_window_end"),
                    of.metadata.get("valid_window_start"),
                    of.metadata.get("valid_window_end"),
                    extra={"policy_tuning_mode": "outer_valid_tail"},
                )
            else:
                for inf in inners:
                    add(
                        "fit_inner_policy",
                        inf.train_idx,
                        inf.valid_idx,
                        inf.inner_fold_id,
                        "inner_train",
                        "inner_oof_valid",
                        inf.metadata.get("train_window_start"),
                        inf.metadata.get("train_window_end"),
                        inf.metadata.get("valid_window_start"),
                        inf.metadata.get("valid_window_end"),
                        extra={"policy_tuning_mode": "inner_oof"},
                    )

        elif consumer_role == "policy_optimiser":
            n_test = len(of.test_idx)
            policy_idx = of.test_idx
            add(
                "predict_policy_oos",
                of.train_idx,
                policy_idx,
                None,
                "full_outer_train",
                "outer_test",
                of.metadata.get("train_window_start"),
                of.metadata.get("train_window_end"),
                of.metadata.get("test_window_start"),
                of.metadata.get("test_window_end"),
                extra={
                    "policy_optimiser_predict_scope": "all_outer_test_rows_not_fit_by_train_base_or_train_meta",
                    "policy_optimiser_holdout_frac": 1.0,
                    "legacy_policy_optimiser_holdout_frac_config": config.policy_optimiser_holdout_frac,
                    "n_policy_predict_rows": int(n_test),
                },
            )

        elif consumer_role == "backtest_eval":
            add(
                "predict_outer_test",
                of.train_idx,
                of.test_idx,
                None,
                "full_outer_train",
                "outer_test",
                of.metadata.get("train_window_start"),
                of.metadata.get("train_window_end"),
                of.metadata.get("test_window_start"),
                of.metadata.get("test_window_end"),
                extra={
                    "contiguous_test_block": True,
                    "test_block_start": of.metadata["test_window_start"],
                    "test_block_end": of.metadata["test_window_end"],
                },
            )

    # Summary log: total rows sent to this consumer
    if plans:
        total_fit_rows = sum(int(p.fit_idx.size) for p in plans)
        total_pred_rows = sum(int(p.predict_idx.size) for p in plans)
        all_symbols = set()
        for p in plans:
            all_symbols.update(p.symbols_fit)
            all_symbols.update(p.symbols_predict)

        # Calculate overall period
        all_fit_starts = [
            p.metadata.get("fit_actual_start")
            for p in plans
            if p.metadata.get("fit_actual_start")
        ]
        all_fit_ends = [
            p.metadata.get("fit_actual_end")
            for p in plans
            if p.metadata.get("fit_actual_end")
        ]
        all_pred_starts = [
            p.metadata.get("predict_actual_start")
            for p in plans
            if p.metadata.get("predict_actual_start")
        ]
        all_pred_ends = [
            p.metadata.get("predict_actual_end")
            for p in plans
            if p.metadata.get("predict_actual_end")
        ]

        overall_start = (
            min(all_fit_starts + all_pred_starts)
            if (all_fit_starts + all_pred_starts)
            else None
        )
        overall_end = (
            max(all_fit_ends + all_pred_ends)
            if (all_fit_ends + all_pred_ends)
            else None
        )

        log(
            f"[slice_planner] SUMMARY for consumer={consumer_role}: "
            f"period={overall_start} to {overall_end}, "
            f"symbols={len(all_symbols)}, "
            f"total_fit_rows={total_fit_rows}, "
            f"total_pred_rows={total_pred_rows}, "
            f"total_rows={total_fit_rows + total_pred_rows}, "
            f"n_plans={len(plans)}"
        )

    return plans


class SlicePlanner:
    """High-level planner building leakage-safe, consumer-aware slice plans."""

    def __init__(self, config: SlicePlannerConfig) -> None:
        self.config = config
        self._log = _default_tprint(config)

    def build(self, events: pd.DataFrame) -> dict[str, Any]:
        clean = validate_events(events, self.config.schema, self.config, self._log)
        state = build_planner_state(clean, self.config)

        outer = make_outer_walkforward_folds(state, self.config, self._log)
        inner_by_outer = {
            of.outer_fold_id: make_inner_purged_folds(state, of, self.config, self._log)
            for of in outer
            if not of.skipped
        }

        consumer = {
            role: build_consumer_slice_plan(
                state, outer, inner_by_outer, role, self.config, self._log
            )
            for role in CONSUMER_ROLES
        }

        # Overall summary across all consumers
        self._log(
            f"[slice_planner] OVERALL SUMMARY: "
            f"preset={self.config.preset.preset_name}, "
            f"n_events={clean.shape[0]}, "
            f"n_symbols={clean[self.config.schema.symbol_col].nunique()}, "
            f"n_outer_folds={len(outer)}, "
            f"n_outer_active={sum(0 if of.skipped else 1 for of in outer)}, "
            f"n_consumers={len(CONSUMER_ROLES)}"
        )

        return {
            "events": clean,
            "state": state,
            "outer_folds": outer,
            "inner_folds_by_outer": inner_by_outer,
            "consumer_plans": consumer,
            "metadata": {
                "preset": self.config.preset.preset_name,
                "n_events": int(clean.shape[0]),
                "n_symbols": int(clean[self.config.schema.symbol_col].nunique()),
                "n_outer_total": len(outer),
                "n_outer_active": int(sum(0 if of.skipped else 1 for of in outer)),
                "symbol_policy_mode": self.config.preset.symbol_policy.mode,
                "sampling_mode": self.config.preset.sampling.mode,
                "purge_scope": self.config.preset.purge_policy.purge_scope,
            },
        }


def build_all_consumer_slices(
    events: pd.DataFrame, config: SlicePlannerConfig
) -> dict[str, list[ConsumerSlicePlan]]:
    """Build once and return all consumer plans to avoid recomputation."""
    return SlicePlanner(config).build(events)["consumer_plans"]


def build_mask_optimiser_slices(
    events: pd.DataFrame,
    config: SlicePlannerConfig,
    plans: Optional[dict[str, list[ConsumerSlicePlan]]] = None,
) -> list[ConsumerSlicePlan]:
    all_plans = plans or build_all_consumer_slices(events, config)
    return all_plans["regime_search"]


def build_tbm_comparison_slices(
    events: pd.DataFrame,
    config: SlicePlannerConfig,
    plans: Optional[dict[str, list[ConsumerSlicePlan]]] = None,
) -> list[ConsumerSlicePlan]:
    all_plans = plans or build_all_consumer_slices(events, config)
    return all_plans["barrier_search"]


def build_base_training_slices(
    events: pd.DataFrame,
    config: SlicePlannerConfig,
    plans: Optional[dict[str, list[ConsumerSlicePlan]]] = None,
) -> list[ConsumerSlicePlan]:
    all_plans = plans or build_all_consumer_slices(events, config)
    return all_plans["base_model_fit"]


def build_meta_training_slices(
    events: pd.DataFrame,
    config: SlicePlannerConfig,
    plans: Optional[dict[str, list[ConsumerSlicePlan]]] = None,
) -> list[ConsumerSlicePlan]:
    all_plans = plans or build_all_consumer_slices(events, config)
    return all_plans["meta_model_fit"]


def build_position_sizer_slices(
    events: pd.DataFrame,
    config: SlicePlannerConfig,
    plans: Optional[dict[str, list[ConsumerSlicePlan]]] = None,
) -> list[ConsumerSlicePlan]:
    all_plans = plans or build_all_consumer_slices(events, config)
    return all_plans["ridge_sizer_fit"]


def build_optimisation_slices(
    events: pd.DataFrame,
    config: SlicePlannerConfig,
    plans: Optional[dict[str, list[ConsumerSlicePlan]]] = None,
) -> list[ConsumerSlicePlan]:
    all_plans = plans or build_all_consumer_slices(events, config)
    return all_plans["utility_policy_tuning"]


def build_policy_optimiser_slices(
    events: pd.DataFrame,
    config: SlicePlannerConfig,
    plans: Optional[dict[str, list[ConsumerSlicePlan]]] = None,
) -> list[ConsumerSlicePlan]:
    all_plans = plans or build_all_consumer_slices(events, config)
    return all_plans["policy_optimiser"]


def build_full_inference_slices(
    events: pd.DataFrame,
    config: SlicePlannerConfig,
    plans: Optional[dict[str, list[ConsumerSlicePlan]]] = None,
) -> list[ConsumerSlicePlan]:
    all_plans = plans or build_all_consumer_slices(events, config)
    return all_plans["full_inference_fit"]


def build_backtest_slices(
    events: pd.DataFrame,
    config: SlicePlannerConfig,
    plans: Optional[dict[str, list[ConsumerSlicePlan]]] = None,
) -> list[ConsumerSlicePlan]:
    all_plans = plans or build_all_consumer_slices(events, config)
    return all_plans["backtest_eval"]


def _demo_events(n: int = 5000, n_symbols: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    start = pd.Timestamp("2021-01-01", tz="UTC")
    t0 = start + pd.to_timedelta(rng.integers(0, 1200, size=n), unit="D")
    hold = rng.integers(1, 20, size=n)
    symbols = np.array([f"SYM{i:03d}" for i in range(n_symbols)])
    sectors = np.array(["tech", "fin", "energy", "health", "utilities"])
    return pd.DataFrame(
        {
            "event_id": np.arange(n, dtype=np.int64),
            "symbol": rng.choice(symbols, size=n),
            "sector": rng.choice(sectors, size=n),
            "t0": t0,
            "t1": t0 + pd.to_timedelta(hold, unit="D"),
        }
    )


if __name__ == "__main__":
    demo = _demo_events()

    robust_cfg = SlicePlannerConfig.robust_defaults(
        schema=EventSchema(group_cols=("sector",))
    )
    robust = SlicePlanner(robust_cfg).build(demo)
    print("robust metadata:", robust["metadata"])

    fast_cfg = SlicePlannerConfig.fast_defaults(
        schema=EventSchema(group_cols=("sector",))
    )
    fast = SlicePlanner(fast_cfg).build(demo)
    print("fast metadata:", fast["metadata"])

    disjoint_cfg = replace(
        robust_cfg,
        preset=replace(
            robust_cfg.preset,
            symbol_policy=replace(
                robust_cfg.preset.symbol_policy,
                mode="disjoint_symbol_holdout",
                unseen_test_symbols=True,
            ),
        ),
    )
    disjoint = SlicePlanner(disjoint_cfg).build(demo)
    print("disjoint metadata:", disjoint["metadata"])

    policy_valid_cfg = replace(
        robust_cfg,
        consumer_overrides={"policy_tuning_mode": "outer_valid_tail"},
    )
    policy_valid = SlicePlanner(policy_valid_cfg).build(demo)

    print("base slices:", len(robust["consumer_plans"]["base_model_fit"]))
    print("meta slices:", len(robust["consumer_plans"]["meta_model_fit"]))
    print("sizer slices:", len(robust["consumer_plans"]["ridge_sizer_fit"]))
    print(
        "policy(inner oof) slices:",
        len(robust["consumer_plans"]["utility_policy_tuning"]),
    )
    print(
        "policy(valid tail) slices:",
        len(policy_valid["consumer_plans"]["utility_policy_tuning"]),
    )
    print("mask optimiser slices:", len(robust["consumer_plans"]["regime_search"]))
    print("tbm slices:", len(robust["consumer_plans"]["barrier_search"]))
    print("backtest slices:", len(robust["consumer_plans"]["backtest_eval"]))
