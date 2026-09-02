#!/usr/bin/env python3
"""Materialise exact Pack-B TP6/SL4/H12 and R3 labels by month and side.

This relabeler is deliberately independent of Pack-B's historical target
shards: those shards supply the immutable candidate population only.  For
every candidate it reopens Kraken one-minute OHLC, derives a decision-time
Wilder ATR(14), and evaluates one fixed contract:

* ``__ts__`` is the completed signal close;
* entry is the exact one-minute open at ``__ts__ + 1h``;
* TP=+6 ATR, SL=-4 ATR, H12 timeout, and an adverse same-minute tie wins;
* gross is the first-touch/timeout exit, and net is gross minus 100 bps once;
* R3 uses the maximum favourable excursion strictly before the first -4 ATR
  adverse touch, with cost +25 bps as the robust-clear hurdle.

The output is checkpointed as ``parts/month=YYYY-MM/side=SIDE.parquet``.
Each completed cell is immutable and can be reused with ``--resume``.  A
partial run has truthful ``manifest.json``/``run_manifest.json`` and
``coverage.parquet``; only a run with every requested month×side cell
complete receives ``status=complete``.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_full_universe_t2_t4_panel import _atr
from scripts.materialize_full_universe_tp6_sl4_h12_sidecar import (
    HORIZON_MINUTES,
    ONE_MINUTE,
    ROUND_TRIP_COST_BPS,
    _first_touch_tp6_sl4,
    _minute_path,
)
from scripts.materialize_tp6_sl4_robust_clear_labels import _pre_adverse_mfe, _sigmoid


SOURCE_DEFAULT = (
    ROOT
    / "data_perp/artifacts/20260720_s59_h5_fullthroughjul10_candleclose_trailing_cost100bps_labels/labels"
)
HISTORICAL_SOURCE_DEFAULT = (
    ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_label_inputs_20260730_v2/candidates.parquet"
)
SIDES = ("long", "short")
START_DEFAULT = "2025-01-01"
END_DEFAULT = "2026-08-01"  # exclusive; covers Jan-2025 through Jul-2026
HISTORICAL_START_DEFAULT = "2022-08-01"
HISTORICAL_END_DEFAULT = "2024-01-01"  # exclusive; Aug-2022 through Dec-2023
TP_ATR = 6.0
SL_ATR = 4.0
COST_BPS = float(ROUND_TRIP_COST_BPS)
ROBUST_BUFFER_BPS = 25.0
ROBUST_TEMPERATURE_BPS = 50.0
# Version three adds the four exact nested first-touch fields used by the
# downstream T3 TBM arm.  A version-two artifact may still be internally
# consistent for TP6/SL4/R3, but it is not a valid source for the six-cell
# broad-to-tail comparison and must never be silently resumed as one.
SCHEMA = "exact_tp6_sl4_h12_r3_relabel_v3"

IDENTITY_COLUMNS = ("candidate_id", "__ts__", "__symbol__", "side_name")
OUTPUT_COLUMNS = (
    *IDENTITY_COLUMNS,
    "__decision_ts__",
    "__label_available_at__",
    "kraken_minute_symbol",
    "tp6_sl4_entry_price",
    "atr_1h",
    "atr_bps",
    "label_valid",
    "target_invalid",
    "invalid_reason",
    # Names deliberately match the 2024 exact-label sidecar so the Stage-I
    # surface can consume Pack-B without an adapter or an ambiguous relabel.
    "t2_tp6_sl4_event",
    "t2_tp6_sl4_exit_minute",
    # Exact nested first-touch minutes for the tail-base TBM target.  These
    # are realised labels only and never inference features.
    "first_tp4_minute",
    "first_tp6_minute",
    "first_sl4_minute",
    "first_sl6_minute",
    "t4_tp6_sl4_exit_pnl_atr",
    "t4_tp6_sl4_terminal_pnl_atr",
    "t4_tp6_sl4_gross_bps",
    "t4_tp6_sl4_net_bps",
    "pre_adverse_mfe_atr",
    "pre_adverse_mfe_bps",
    "lower_touch_minute",
    "robust_clear_margin_bps_b25",
    "robust_clear_event_b25",
    "robust_clear_soft_b25_t50",
)


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    out = pd.Timestamp(value)
    if out.tzinfo is None:
        out = out.tz_localize("UTC")
    else:
        out = out.tz_convert("UTC")
    return out


def _months(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    if start != start.normalize().replace(day=1) or end != end.normalize().replace(day=1):
        raise ValueError("start/end must be UTC first-of-month boundaries")
    return list(pd.date_range(start, end - pd.offsets.MonthBegin(1), freq="MS", tz="UTC"))


def _packb_to_kraken_symbol(symbol: str) -> str:
    """Translate Pack-B's slash symbol to the execution_1m partition name."""
    value = str(symbol).strip()
    if not value:
        raise ValueError("blank Pack-B symbol")
    return value.replace("/", "_")


def _overlapping_minute_fragments(
    root: Path, symbol: str, start: pd.Timestamp, end_exclusive: pd.Timestamp
) -> list[Path]:
    """Use immutable filename epoch bounds to avoid scanning every fragment."""
    location = root / f"symbol={symbol}"
    if not location.exists():
        return []
    start_epoch = int(start.timestamp())
    end_epoch = int(end_exclusive.timestamp())
    years = range(start.year, (end_exclusive - pd.Timedelta(minutes=1)).year + 1)
    selected: list[Path] = []
    for year in years:
        for path in sorted((location / f"year={year}").glob("*.parquet")):
            tokens = path.stem.rsplit("-", 2)
            try:
                fragment_start, fragment_end = int(tokens[-2]), int(tokens[-1])
            except (ValueError, IndexError):
                # A legacy filename without declared bounds cannot be safely
                # skipped; retain it and let the exact timestamp filter decide.
                selected.append(path)
                continue
            if fragment_end >= start_epoch and fragment_start < end_epoch:
                selected.append(path)
    return selected


def _minute_path_pruned(
    root: Path, symbol: str, start: pd.Timestamp, end_exclusive: pd.Timestamp
) -> pd.DataFrame:
    """Read only overlapping immutable fragments, then enforce an exact grid."""
    fragments = _overlapping_minute_fragments(root, symbol, start, end_exclusive)
    tables: list[pa.Table] = []
    unreadable: list[str] = []
    for path in fragments:
        try:
            tables.append(pq.ParquetFile(path).read(columns=["ts", "open", "high", "low", "close"]))
        except (OSError, pa.ArrowInvalid) as exc:
            # Never substitute an alternate bar or forward-fill an unreadable
            # source fragment.  Leaving the grid empty at its timestamps makes
            # only affected H12 paths / causal ATRs invalid downstream.  The
            # audit is retained on the frame for the caller's run manifest.
            unreadable.append(f"{path.name}: {type(exc).__name__}")
    if tables:
        # Historical fragments were written by more than one downloader and
        # may store OHLC as float32 or float64.  Permissive promotion preserves
        # values while keeping the timestamp filter exact.
        raw = pa.concat_tables(tables, promote_options="permissive").to_pandas()
        raw["ts"] = pd.to_datetime(raw["ts"], utc=True, errors="raise")
        raw = raw.loc[raw["ts"].ge(start) & raw["ts"].lt(end_exclusive)]
        raw = raw.drop_duplicates("ts", keep="last").set_index("ts").sort_index()
    else:
        raw = pd.DataFrame(columns=["open", "high", "low", "close"], index=pd.DatetimeIndex([], tz="UTC"))
    grid = pd.date_range(
        start.floor("min"), (end_exclusive - pd.Timedelta(minutes=1)).floor("min"),
        freq="min", tz="UTC",
    )
    result = raw.reindex(grid)
    result.attrs["unreadable_minute_fragments"] = tuple(unreadable)
    return result


def _source_path(source: Path, month: pd.Timestamp, side: str) -> Path:
    return source / f"train_global_{side}_5_{month:%Y_%m}.parquet"


def _validate_candidate_frame(
    frame: pd.DataFrame, *, month: pd.Timestamp, side: str, source_kind: str, source_name: Path
) -> pd.DataFrame:
    """Validate immutable candidate identity without assuming an ID encoding.

    Pack-B candidate IDs are transparent signal/side strings and are checked
    exactly.  The historical ledger uses stable hashes, so its immutable full
    identity is the candidate id plus timestamp, symbol and side; treating a
    hash as though it had Pack-B's spelling would reject valid older data.
    """
    missing = sorted(set(IDENTITY_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"candidate source lacks identity fields {missing}: {source_name}")
    frame = frame.loc[:, list(IDENTITY_COLUMNS)].copy()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    if frame.empty:
        raise ValueError(f"candidate source is empty: {source_name}")
    if frame["candidate_id"].isna().any() or frame["candidate_id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"candidate source has blank identities: {source_name}")
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"candidate source has duplicate candidate ids: {source_name}")
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError(f"candidate source has duplicate full identities: {source_name}")
    if not frame["side_name"].eq(side).all():
        raise ValueError(f"candidate source is not side-local ({side}): {source_name}")
    expected_end = month + pd.offsets.MonthBegin(1)
    if not frame["__ts__"].between(month, expected_end - pd.Timedelta(nanoseconds=1)).all():
        raise ValueError(f"candidate source escapes its month: {source_name}")
    if source_kind == "packb":
        # Pack-B uses a transparent identity, but historical shards spell UTC
        # with ``Z`` while pandas' ``isoformat`` uses ``+00:00``.  Parse the
        # four components and compare their semantics instead of requiring one
        # timestamp spelling.  ``rsplit`` preserves symbols containing ``|``
        # should the exchange ever expose one.
        parsed = frame["candidate_id"].astype(str).str.rsplit("|", n=3, expand=True)
        if parsed.shape[1] != 4:
            raise ValueError(f"Pack-B candidate identity does not bind symbol/signal/side: {source_name}")
        parsed_ts = pd.to_datetime(parsed[1], utc=True, errors="coerce")
        identity_ok = (
            parsed[0].eq(frame["__symbol__"].astype(str))
            & parsed_ts.eq(frame["__ts__"])
            & parsed[2].eq("1h")
            & parsed[3].eq(frame["side_name"].astype(str))
        )
        if not identity_ok.all():
            raise ValueError(f"Pack-B candidate identity does not bind symbol/signal/side: {source_name}")
    return frame.sort_values(["__symbol__", "__ts__", "candidate_id"], kind="mergesort").reset_index(drop=True)


def _load_packb_candidates(
    source: Path, month: pd.Timestamp, side: str, *, symbols: set[str] | None = None
) -> pd.DataFrame:
    path = _source_path(source, month, side)
    if not path.exists():
        raise FileNotFoundError(f"Pack-B candidate shard absent: {path}")
    frame = pd.read_parquet(path, columns=list(IDENTITY_COLUMNS))
    if symbols is not None:
        frame = frame.loc[frame["__symbol__"].astype(str).isin(symbols)].copy()
        if frame.empty:
            raise ValueError(f"no candidates remain after symbol filter for {path}")
    return _validate_candidate_frame(
        frame,
        month=month, side=side, source_kind="packb", source_name=path,
    )


def _load_historical_candidates(source: Path, month: pd.Timestamp, side: str) -> pd.DataFrame:
    if not source.exists():
        raise FileNotFoundError(f"historical candidate ledger absent: {source}")
    frame = pd.read_parquet(source, columns=list(IDENTITY_COLUMNS))
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    next_month = month + pd.offsets.MonthBegin(1)
    frame = frame.loc[frame["side_name"].eq(side) & ts.between(month, next_month - pd.Timedelta(nanoseconds=1))]
    return _validate_candidate_frame(
        frame, month=month, side=side, source_kind="historical", source_name=source,
    )


def _load_stage_i_common30_candidates(
    source: Path, month: pd.Timestamp, side: str
) -> pd.DataFrame:
    """Load the immutable product-bound Stage-I request population.

    The request stage deliberately uses public names (``signal_timestamp`` and
    ``symbol``).  Convert those names once at the label boundary, while
    preserving its frozen candidate id and exact side/month population.
    """
    path = source / "staged_candidates.parquet" if source.is_dir() else source
    if not path.exists():
        raise FileNotFoundError(f"Stage-I common30 candidate ledger absent: {path}")
    columns = ["candidate_id", "signal_timestamp", "symbol", "side_name"]
    frame = pd.read_parquet(path, columns=columns).rename(
        columns={"signal_timestamp": "__ts__", "symbol": "__symbol__"}
    )
    ts = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    next_month = month + pd.offsets.MonthBegin(1)
    frame = frame.loc[
        frame["side_name"].eq(side)
        & ts.between(month, next_month - pd.Timedelta(nanoseconds=1))
    ]
    return _validate_candidate_frame(
        frame, month=month, side=side,
        source_kind="stage_i_common30", source_name=path,
    )


def _load_candidates(
    source: Path, month: pd.Timestamp, side: str, *, source_kind: str,
    symbols: set[str] | None = None,
) -> pd.DataFrame:
    if source_kind == "packb":
        return _load_packb_candidates(source, month, side, symbols=symbols)
    if source_kind == "historical":
        return _load_historical_candidates(source, month, side)
    if source_kind == "stage_i_common30":
        return _load_stage_i_common30_candidates(source, month, side)
    if source_kind == "generic":
        return _load_historical_candidates(source, month, side)
    raise ValueError(f"unsupported candidate source kind: {source_kind}")


def _causal_hourly_atr_from_minute(minute: pd.DataFrame) -> pd.Series:
    """Return ATR(14) indexed by completed hourly candle close timestamps.

    The minute interval ``[t-1h, t)`` is labelled at ``t``.  Therefore ATR at
    a Pack-B signal close ``t`` never reads the entry bar at ``t+1h`` or any
    future minute.  The Wilder calculation itself is the canonical ``_atr``
    implementation used by the full-universe label materialiser.
    """
    # An hourly candle is eligible only when *every* minute OHLC observation
    # is finite.  Resampling a partial hour would silently turn missing path
    # data into an apparently valid decision-time ATR.  Requiring 14
    # consecutive complete input candles is stricter than the canonical
    # normal-data path while preserving its values whenever the source is
    # complete (the production case).
    ohlc = minute.loc[:, ["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    minute_complete = pd.Series(
        np.isfinite(ohlc.to_numpy(dtype=np.float64)).all(axis=1), index=minute.index
    )
    hourly = ohlc.resample("1h", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")
    )
    complete_hour = minute_complete.resample("1h", label="left", closed="left").sum().eq(60)
    hourly.loc[~complete_hour, :] = np.nan
    hourly.index = hourly.index + pd.Timedelta(hours=1)
    complete_hour.index = hourly.index
    atr = _atr(hourly, period=14)
    return atr.where(complete_hour.rolling(14, min_periods=14).sum().eq(14))


def _complete_h12_paths(minute: pd.DataFrame, starts: np.ndarray) -> np.ndarray:
    finite = np.isfinite(minute[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)).all(axis=1)
    cumulative = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    out = np.zeros(len(starts), dtype=bool)
    in_bounds = (starts >= 0) & (starts + HORIZON_MINUTES <= len(minute))
    valid_starts = starts[in_bounds]
    out[in_bounds] = (cumulative[valid_starts + HORIZON_MINUTES] - cumulative[valid_starts]) == HORIZON_MINUTES
    return out


@njit(cache=True)
def _nested_tbm_first_touch_minutes(
    high: np.ndarray,
    low: np.ndarray,
    starts: np.ndarray,
    entry: np.ndarray,
    atr: np.ndarray,
    side: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Record exact H12 first touches for +/-4 and +/-6 ATR barriers.

    A result is a zero-based minute offset or ``-1`` for a valid no-touch.
    This routine intentionally records the physical crossings independently;
    the later five-grade TBM target applies its declared adverse tie break.
    """
    n = len(starts)
    tp4 = np.full(n, -1, np.int16)
    tp6 = np.full(n, -1, np.int16)
    sl4 = np.full(n, -1, np.int16)
    sl6 = np.full(n, -1, np.int16)
    for row in range(n):
        start = starts[row]
        e = entry[row]
        a = atr[row]
        if start < 0 or start + HORIZON_MINUTES > len(high) or not np.isfinite(e) or not np.isfinite(a) or a <= 0.0:
            continue
        direction = side[row]
        complete = True
        for offset in range(HORIZON_MINUTES):
            pos = start + offset
            if not np.isfinite(high[pos]) or not np.isfinite(low[pos]):
                complete = False
                break
            if direction > 0.0:
                favourable = (high[pos] - e) / a
                adverse = (e - low[pos]) / a
            else:
                favourable = (e - low[pos]) / a
                adverse = (high[pos] - e) / a
            if tp4[row] < 0 and favourable >= 4.0:
                tp4[row] = offset
            if tp6[row] < 0 and favourable >= 6.0:
                tp6[row] = offset
            if sl4[row] < 0 and adverse >= 4.0:
                sl4[row] = offset
            if sl6[row] < 0 and adverse >= 6.0:
                sl6[row] = offset
        if not complete:
            tp4[row] = -1
            tp6[row] = -1
            sl4[row] = -1
            sl6[row] = -1
    return tp4, tp6, sl4, sl6


def _invalid_reason(
    *,
    identity_ok: np.ndarray,
    side_ok: np.ndarray,
    exact_entry: np.ndarray,
    atr_ok: np.ndarray,
    complete: np.ndarray,
    primitive_valid: np.ndarray,
    economics_ok: np.ndarray,
) -> np.ndarray:
    result = np.full(len(identity_ok), "complete_executable_h12_path", dtype=object)
    result[~economics_ok] = "unresolved_exact_economic_path"
    result[~primitive_valid] = "unresolved_robust_clear_path"
    result[~complete] = "incomplete_h12_ohlc_path"
    result[~atr_ok] = "missing_or_nonpositive_causal_atr14"
    result[~exact_entry] = "missing_or_nonfinite_exact_entry_open"
    result[~side_ok] = "invalid_side"
    result[~identity_ok] = "missing_or_invalid_candidate_identity"
    return result


def _label_candidates_with_minute(
    candidates: pd.DataFrame, minute: pd.DataFrame, *, atr_hourly: pd.Series | None = None
) -> pd.DataFrame:
    """Label one same-symbol candidate frame against its already-loaded path.

    Kept public-by-convention for focused synthetic tests.  It does no IO and
    is the precise common path used by the month×side driver.
    """
    x = candidates.copy().reset_index(drop=True)
    x["__ts__"] = pd.to_datetime(x["__ts__"], utc=True, errors="raise")
    decision = x["__ts__"] + pd.Timedelta(hours=1)
    starts = minute.index.get_indexer(pd.DatetimeIndex(decision)).astype(np.int64)
    entry = np.full(len(x), np.nan, dtype=np.float64)
    exact_entry = starts >= 0
    if exact_entry.any():
        entry[exact_entry] = minute["open"].to_numpy(dtype=np.float64)[starts[exact_entry]]
    if atr_hourly is None:
        atr_hourly = _causal_hourly_atr_from_minute(minute)
    atr = atr_hourly.reindex(pd.DatetimeIndex(x["__ts__"])).to_numpy(dtype=np.float64)
    atr_ok = np.isfinite(atr) & (atr > 0.0)
    complete = _complete_h12_paths(minute, starts)
    side_ok = x["side_name"].isin(SIDES).to_numpy(dtype=bool)
    side = np.where(x["side_name"].eq("long").to_numpy(), 1.0, -1.0)
    event, exit_minute, exit_pnl_atr, terminal_pnl_atr = _first_touch_tp6_sl4(
        minute["high"].to_numpy(dtype=np.float64),
        minute["low"].to_numpy(dtype=np.float64),
        minute["close"].to_numpy(dtype=np.float64),
        starts,
        entry,
        atr,
        side,
    )
    pre_valid, pre_mfe_atr, lower_touch = _pre_adverse_mfe(
        minute["high"].to_numpy(dtype=np.float64),
        minute["low"].to_numpy(dtype=np.float64),
        minute["close"].to_numpy(dtype=np.float64),
        starts,
        entry,
        atr,
        side,
    )
    first_tp4, first_tp6, first_sl4, first_sl6 = _nested_tbm_first_touch_minutes(
        minute["high"].to_numpy(dtype=np.float64), minute["low"].to_numpy(dtype=np.float64),
        starts, entry, atr, side,
    )
    economics_ok = np.isfinite(exit_pnl_atr) & np.isfinite(terminal_pnl_atr)
    label_valid = exact_entry & atr_ok & complete & pre_valid & economics_ok & side_ok
    identity_ok = (
        x["candidate_id"].notna().to_numpy(dtype=bool)
        & x["candidate_id"].astype(str).str.strip().ne("").to_numpy(dtype=bool)
        & x["__symbol__"].notna().to_numpy(dtype=bool)
    )
    label_valid &= identity_ok
    gross = exit_pnl_atr.astype(np.float64) * atr / entry * 10_000.0
    net = gross - COST_BPS
    atr_bps = atr / entry * 10_000.0
    pre_mfe_bps = pre_mfe_atr.astype(np.float64) * atr_bps
    robust_margin = pre_mfe_bps - COST_BPS - ROBUST_BUFFER_BPS
    robust_event = robust_margin > 0.0
    out = x.loc[:, list(IDENTITY_COLUMNS)].copy()
    out["__decision_ts__"] = decision
    out["__label_available_at__"] = decision + pd.Timedelta(hours=12)
    out["kraken_minute_symbol"] = out["__symbol__"].map(_packb_to_kraken_symbol)
    out["tp6_sl4_entry_price"] = entry
    out["atr_1h"] = np.where(label_valid, atr, np.nan)
    out["atr_bps"] = np.where(label_valid, atr_bps, np.nan)
    out["label_valid"] = label_valid
    out["target_invalid"] = ~label_valid
    out["invalid_reason"] = _invalid_reason(
        identity_ok=identity_ok,
        side_ok=side_ok,
        exact_entry=exact_entry,
        atr_ok=atr_ok,
        complete=complete,
        primitive_valid=pre_valid,
        economics_ok=economics_ok,
    )
    out.loc[label_valid, "invalid_reason"] = "complete_executable_h12_path"
    out["t2_tp6_sl4_event"] = np.where(label_valid, event, np.nan)
    out["t2_tp6_sl4_exit_minute"] = np.where(label_valid, exit_minute, np.nan)
    out["first_tp4_minute"] = np.where(label_valid, first_tp4, np.nan)
    out["first_tp6_minute"] = np.where(label_valid, first_tp6, np.nan)
    out["first_sl4_minute"] = np.where(label_valid, first_sl4, np.nan)
    out["first_sl6_minute"] = np.where(label_valid, first_sl6, np.nan)
    out["t4_tp6_sl4_exit_pnl_atr"] = np.where(label_valid, exit_pnl_atr, np.nan)
    out["t4_tp6_sl4_terminal_pnl_atr"] = np.where(label_valid, terminal_pnl_atr, np.nan)
    out["t4_tp6_sl4_gross_bps"] = np.where(label_valid, gross, np.nan)
    out["t4_tp6_sl4_net_bps"] = np.where(label_valid, net, np.nan)
    out["pre_adverse_mfe_atr"] = np.where(label_valid, pre_mfe_atr, np.nan)
    out["pre_adverse_mfe_bps"] = np.where(label_valid, pre_mfe_bps, np.nan)
    out["lower_touch_minute"] = np.where(label_valid, lower_touch, np.nan)
    out["robust_clear_margin_bps_b25"] = np.where(label_valid, robust_margin, np.nan)
    out["robust_clear_event_b25"] = np.where(label_valid, robust_event.astype(np.float32), np.nan)
    out["robust_clear_soft_b25_t50"] = np.where(
        label_valid, _sigmoid(robust_margin / ROBUST_TEMPERATURE_BPS), np.nan
    )
    if not np.allclose(
        out.loc[label_valid, "t4_tp6_sl4_gross_bps"].to_numpy(dtype=float) - COST_BPS,
        out.loc[label_valid, "t4_tp6_sl4_net_bps"].to_numpy(dtype=float),
        rtol=0.0,
        atol=2e-3,
    ):
        raise AssertionError("Pack-B TP6/SL4 net cost was not applied exactly once")
    invalid_targets = [
        "t2_tp6_sl4_event", "t2_tp6_sl4_exit_minute", "first_tp4_minute", "first_tp6_minute", "first_sl4_minute", "first_sl6_minute", "t4_tp6_sl4_exit_pnl_atr", "t4_tp6_sl4_terminal_pnl_atr",
        "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "pre_adverse_mfe_atr", "pre_adverse_mfe_bps",
        "lower_touch_minute", "robust_clear_margin_bps_b25", "robust_clear_event_b25", "robust_clear_soft_b25_t50",
    ]
    if out.loc[~label_valid, invalid_targets].notna().any().any():
        raise AssertionError("invalid Pack-B paths acquired an economic target")
    return out.reindex(columns=OUTPUT_COLUMNS)


def _materialise_cell(candidates: pd.DataFrame, minute_root: Path) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for raw_symbol, group in candidates.groupby("__symbol__", sort=True):
        symbol = _packb_to_kraken_symbol(str(raw_symbol))
        start = pd.Timestamp(group["__ts__"].min()) - pd.Timedelta(hours=14)
        end = pd.Timestamp(group["__ts__"].max()) + pd.Timedelta(hours=13)
        minute = _minute_path_pruned(minute_root, symbol, start, end)
        outputs.append(_label_candidates_with_minute(group, minute))
    return pd.concat(outputs, ignore_index=True).sort_values(["__ts__", "__symbol__", "candidate_id"], kind="mergesort").reset_index(drop=True)


def _materialise_month_sides(
    candidates_by_side: dict[str, pd.DataFrame],
    minute_root: Path,
    *,
    minute_loader: Any = _minute_path_pruned,
) -> dict[str, pd.DataFrame]:
    """Materialise missing sides using one minute/ATR substrate per symbol.

    Pack-B long and short candidate streams frequently share the same symbol
    and month.  Loading both separately doubles a large amount of immutable
    minute IO and repeats the identical causal ATR calculation.  The union
    range is safe because each label still reindexes ATR at its own signal
    close and scans only its own exact H12 path.
    """
    if not candidates_by_side:
        return {}
    if set(candidates_by_side) - set(SIDES):
        raise ValueError("month materialisation received an unsupported side")
    outputs: dict[str, list[pd.DataFrame]] = {side: [] for side in candidates_by_side}
    combined = pd.concat(
        [frame.assign(__materialise_side__=side) for side, frame in candidates_by_side.items()],
        ignore_index=True,
    )
    for symbol_index, (raw_symbol, all_symbol_rows) in enumerate(combined.groupby("__symbol__", sort=True), 1):
        start = pd.Timestamp(all_symbol_rows["__ts__"].min()) - pd.Timedelta(hours=14)
        end = pd.Timestamp(all_symbol_rows["__ts__"].max()) + pd.Timedelta(hours=13)
        minute_symbol = _packb_to_kraken_symbol(str(raw_symbol))
        source_exists = (
            (minute_root / f"symbol={minute_symbol}").exists()
            if minute_loader is _minute_path_pruned else True
        )
        minute = minute_loader(minute_root, minute_symbol, start, end)
        atr_hourly = _causal_hourly_atr_from_minute(minute)
        for side, group in all_symbol_rows.groupby("__materialise_side__", sort=False):
            labelled = _label_candidates_with_minute(
                group.drop(columns="__materialise_side__"), minute, atr_hourly=atr_hourly
            )
            if not source_exists:
                if labelled.label_valid.any():
                    raise AssertionError("absent minute source unexpectedly produced a valid label")
                labelled["invalid_reason"] = "symbol_minute_source_unavailable"
            outputs[str(side)].append(labelled)
        # Arrow's global memory pool can retain buffers from every symbol
        # scan.  Across 150+ symbols that turns a small one-month relabel into
        # an avoidable multi-GB resident process.  Release scanner buffers in
        # bounded batches; output frames are small and remain in ``outputs``.
        del minute, atr_hourly
        if symbol_index % 8 == 0:
            gc.collect()
            pa.default_memory_pool().release_unused()
    gc.collect()
    pa.default_memory_pool().release_unused()
    return {
        side: pd.concat(pieces, ignore_index=True)
        .sort_values(["__ts__", "__symbol__", "candidate_id"], kind="mergesort")
        .reset_index(drop=True)
        for side, pieces in outputs.items()
    }


def _coverage_record(path: Path, month: str, side: str, status: str) -> dict[str, Any]:
    if status != "materialised" and not path.exists():
        return {"month": month, "side": side, "status": status, "rows": 0}
    frame = pd.read_parquet(
        path,
        columns=["candidate_id", "label_valid", "target_invalid", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps", "invalid_reason"],
    )
    valid = frame["label_valid"].astype(bool)
    return {
        "month": month,
        "side": side,
        "status": status,
        "rows": int(len(frame)),
        "valid_rows": int(valid.sum()),
        "invalid_rows": int((~valid).sum()),
        "valid_fraction": float(valid.mean()) if len(frame) else np.nan,
        "duplicate_candidate_ids": int(frame["candidate_id"].duplicated().sum()),
        "invalid_target_rows": int(frame.loc[~valid, "t4_tp6_sl4_net_bps"].notna().sum()),
        "gross_net_identity_rows": int(
            np.isclose(
                frame.loc[valid, "t4_tp6_sl4_gross_bps"].to_numpy(dtype=float) - COST_BPS,
                frame.loc[valid, "t4_tp6_sl4_net_bps"].to_numpy(dtype=float),
                rtol=0.0,
                atol=2e-3,
            ).sum()
        ),
        "invalid_reason_counts": json.dumps(
            frame.loc[~valid, "invalid_reason"].value_counts(dropna=False).to_dict(), sort_keys=True
        ),
    }


def _validate_checkpoint(path: Path, expected: pd.DataFrame) -> None:
    # Read the schema before selecting columns so an older TP6/SL4-only
    # checkpoint receives a clear fail-closed diagnostic rather than being
    # treated as a resumable exact-T3 recovery.
    frame = pd.read_parquet(path)
    missing = sorted(set(OUTPUT_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(
            "exact relabel checkpoint lacks the current T3-capable schema "
            f"({missing}); rebuild it into a new immutable output directory: {path}"
        )
    frame = frame.loc[:, list(OUTPUT_COLUMNS)]
    if len(frame) != len(expected) or frame["candidate_id"].duplicated().any():
        raise ValueError(f"invalid exact relabel checkpoint: {path}")
    if set(frame["candidate_id"].astype(str)) != set(expected["candidate_id"].astype(str)):
        raise ValueError(f"exact relabel checkpoint changed candidate identities: {path}")
    if frame.loc[frame.target_invalid.astype(bool), "t4_tp6_sl4_net_bps"].notna().any():
        raise ValueError(f"exact relabel checkpoint encodes invalid rows as economics: {path}")
    valid = frame.label_valid.astype(bool)
    if not np.allclose(
        frame.loc[valid, "t4_tp6_sl4_gross_bps"].to_numpy(dtype=float) - COST_BPS,
        frame.loc[valid, "t4_tp6_sl4_net_bps"].to_numpy(dtype=float),
        rtol=0.0,
        atol=2e-3,
    ):
        raise ValueError(f"exact relabel checkpoint has invalid cost arithmetic: {path}")


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    tmp = Path(raw)
    try:
        frame.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(destination)
    finally:
        if tmp.exists():
            tmp.unlink()


def _symbol_checkpoint_path(root: Path, month: pd.Timestamp, symbol: str, side: str) -> Path:
    token = hashlib.sha256(str(symbol).encode()).hexdigest()[:20]
    return root / "symbol_parts" / f"month={month:%Y-%m}" / f"symbol={token}" / f"side={side}.parquet"


def _materialise_symbol_checkpoint_batch(
    candidates_by_side: dict[str, pd.DataFrame],
    minute_root: Path,
    *,
    month: pd.Timestamp,
    output_root: Path,
    batch_size: int,
    minute_loader: Any = _minute_path_pruned,
) -> dict[str, int]:
    """Materialise at most ``batch_size`` missing symbols, atomically."""
    symbols = sorted(set().union(*(set(frame["__symbol__"].astype(str)) for frame in candidates_by_side.values())))
    pending = [
        symbol for symbol in symbols
        if any(
            not _symbol_checkpoint_path(output_root, month, symbol, side).exists()
            for side, frame in candidates_by_side.items()
            if frame["__symbol__"].astype(str).eq(symbol).any()
        )
    ]
    selected = pending[:batch_size]
    if selected:
        selected_set = set(selected)
        subset = {
            side: frame.loc[frame["__symbol__"].astype(str).isin(selected_set)].copy()
            for side, frame in candidates_by_side.items()
        }
        materialised = _materialise_month_sides(subset, minute_root, minute_loader=minute_loader)
        for side, result in materialised.items():
            for symbol, part in result.groupby("__symbol__", sort=True):
                destination = _symbol_checkpoint_path(output_root, month, str(symbol), side)
                expected = subset[side].loc[subset[side]["__symbol__"].astype(str).eq(str(symbol))]
                if len(part) != len(expected) or set(part.candidate_id) != set(expected.candidate_id):
                    raise ValueError("symbol checkpoint changed candidate identity/cardinality")
                _atomic_parquet(part, destination)
    remaining = [
        symbol for symbol in symbols
        if any(
            not _symbol_checkpoint_path(output_root, month, symbol, side).exists()
            for side, frame in candidates_by_side.items()
            if frame["__symbol__"].astype(str).eq(symbol).any()
        )
    ]
    return {"total_symbols": len(symbols), "processed_this_batch": len(selected), "remaining_symbols": len(remaining)}


def _assemble_symbol_checkpoints(
    candidates_by_side: dict[str, pd.DataFrame], *, month: pd.Timestamp, output_root: Path
) -> dict[str, pd.DataFrame] | None:
    """Assemble a month only after every immutable symbol cell is present."""
    outputs: dict[str, pd.DataFrame] = {}
    for side, expected in candidates_by_side.items():
        paths = [
            _symbol_checkpoint_path(output_root, month, str(symbol), side)
            for symbol in sorted(expected["__symbol__"].astype(str).unique())
        ]
        if any(not path.exists() for path in paths):
            return None
        result = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        result = result.sort_values(["__ts__", "__symbol__", "candidate_id"], kind="mergesort").reset_index(drop=True)
        if len(result) != len(expected) or set(result.candidate_id) != set(expected.candidate_id):
            raise ValueError("assembled symbol checkpoints changed candidate identity/cardinality")
        outputs[side] = result
    return outputs


def _write_status(
    out: Path, records: Iterable[dict[str, Any]], *, source: Path, minute_root: Path,
    start: pd.Timestamp, end: pd.Timestamp, required_months: list[pd.Timestamp],
    required_sides: list[str] | None = None, source_kind: str = "packb",
) -> None:
    # Preserve the historical two-side manifest contract for callers that do
    # not opt into the newer side-local materialisation surface.
    required_sides = list(required_sides or ["long", "short"])
    coverage = pd.DataFrame(list(records))
    if coverage.empty:
        coverage = pd.DataFrame(columns=["month", "side", "status", "rows"])
    else:
        coverage = coverage.sort_values(["month", "side"], kind="mergesort")
    coverage.to_parquet(out / "coverage.parquet", index=False, compression="zstd")
    required_keys = {
        (month.strftime("%Y-%m"), side)
        for month in required_months for side in required_sides
    }
    covered_keys = {
        (str(row.month), str(row.side))
        for row in coverage.itertuples(index=False)
        if str(row.status) in {"materialised", "reused"}
    }
    expected_cells = len(required_keys)
    complete = required_keys.issubset(covered_keys)
    manifest = {
        "schema": SCHEMA,
        "status": "complete" if complete else "partial",
        "complete": bool(complete),
        "candidate_source_kind": source_kind,
        "source_candidates": str(source),
        "minute_root": str(minute_root),
        "range": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "cells": {
            "months": [m.strftime("%Y-%m") for m in required_months],
            "sides": list(required_sides), "expected": expected_cells,
            "complete": int(len(required_keys.intersection(covered_keys))),
        },
        "contract": {
            "entry": "candidate __ts__ signal close +1h; exact decision-minute open",
            "atr": "14 completed hourly candles from minute OHLC; Wilder alpha=1/14; signal-close causal",
            "geometry": "TP +6 ATR / SL -4 ATR / H12",
            "same_minute_conflict": "adverse (SL) precedence",
            "cost_bps": COST_BPS,
            "net_formula": "gross_bps - 100 exactly once",
            "r3": "robust_clear: pre-adverse MFE strictly before -4 ATR touch; robust_clear = pre-adverse MFE bps -100 -25 > 0; sigmoid temperature 50 bps (B25/T50)",
            "invalid_rows": "target_invalid=true and every economic/R3 target is null; retained solely for coverage",
        },
        "coverage_path": str(out / "coverage.parquet"),
    }
    payload = json.dumps(manifest, indent=2) + "\n"
    # ``manifest.json`` is the consumption contract.  Keep the old filename
    # as an identical convenience copy for resumable-job diagnostics.
    (out / "manifest.json").write_text(payload, encoding="utf-8")
    (out / "run_manifest.json").write_text(payload, encoding="utf-8")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-kind", choices=("packb", "historical", "stage_i_common30", "generic"),
        default="packb",
    )
    p.add_argument("--source", type=Path, default=None, help="Pack-B shard directory or historical candidates.parquet")
    p.add_argument("--minute-root", type=Path, default=ONE_MINUTE)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None, help="exclusive month boundary")
    p.add_argument("--month", action="append", default=[], help="optional YYYY-MM checkpoint scope; repeatable")
    p.add_argument("--side", action="append", choices=SIDES, default=[])
    p.add_argument(
        "--symbols-file", type=Path, default=None,
        help="optional CSV containing a symbol column; Pack-B candidates are restricted to this universe",
    )
    p.add_argument(
        "--symbol-batch-size", type=int, default=0,
        help="checkpoint at most N missing symbols for each selected month; rerun with --resume until assembled",
    )
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _args()
    source = args.source or (SOURCE_DEFAULT if args.source_kind == "packb" else HISTORICAL_SOURCE_DEFAULT)
    default_start = START_DEFAULT if args.source_kind == "packb" else HISTORICAL_START_DEFAULT
    default_end = END_DEFAULT if args.source_kind == "packb" else HISTORICAL_END_DEFAULT
    start, end = _utc(args.start or default_start), _utc(args.end or default_end)
    all_months = _months(start, end)
    selected_months = (
        [_utc(f"{value}-01") for value in args.month]
        if args.month else all_months
    )
    if any(value not in all_months for value in selected_months):
        raise ValueError("--month must fall inside the requested source-kind range")
    selected_months = sorted(set(selected_months))
    selected_sides = list(dict.fromkeys(args.side or SIDES))
    symbols: set[str] | None = None
    if args.symbols_file is not None:
        symbol_frame = pd.read_csv(args.symbols_file)
        if "symbol" not in symbol_frame.columns:
            raise ValueError(f"symbols file must contain a symbol column: {args.symbols_file}")
        symbols = {
            str(value).strip() for value in symbol_frame["symbol"].dropna()
            if str(value).strip()
        }
        if not symbols:
            raise ValueError(f"symbols file contains no symbols: {args.symbols_file}")
    if args.symbol_batch_size < 0:
        raise ValueError("--symbol-batch-size must be non-negative")
    if args.out.exists() and not args.resume:
        raise FileExistsError(f"output exists: {args.out}; use --resume")
    args.out.mkdir(parents=True, exist_ok=True)
    # Reconstruct coverage from every existing cell before working on the
    # requested subset.  A one-month invocation must not erase the evidence
    # from cells completed by a previous invocation.
    records_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for month in all_months:
        for side in SIDES:
            destination = args.out / "parts" / f"month={month:%Y-%m}" / f"side={side}.parquet"
            if destination.exists():
                record = _coverage_record(destination, f"{month:%Y-%m}", side, "reused")
                records_by_key[(f"{month:%Y-%m}", side)] = record
    for month in selected_months:
        pending: dict[str, pd.DataFrame] = {}
        destinations: dict[str, Path] = {}
        for side in selected_sides:
            candidates = _load_candidates(
                source, month, side, source_kind=args.source_kind, symbols=symbols
            )
            destination = args.out / "parts" / f"month={month:%Y-%m}" / f"side={side}.parquet"
            destinations[side] = destination
            if destination.exists():
                if not args.resume:
                    raise FileExistsError(f"checkpoint exists: {destination}")
                _validate_checkpoint(destination, candidates)
                records_by_key[(f"{month:%Y-%m}", side)] = _coverage_record(
                    destination, f"{month:%Y-%m}", side, "reused"
                )
            else:
                pending[side] = candidates

        # One shared symbol-wise source pass for every missing side in this
        # month.  Side files are still written and checkpointed independently.
        if pending and args.symbol_batch_size:
            progress = _materialise_symbol_checkpoint_batch(
                pending, args.minute_root, month=month, output_root=args.out,
                batch_size=args.symbol_batch_size,
            )
            materialised = _assemble_symbol_checkpoints(
                pending, month=month, output_root=args.out,
            )
            if materialised is None:
                _write_status(
                    args.out, records_by_key.values(), source=source, minute_root=args.minute_root,
                    start=start, end=end, required_months=all_months,
                    required_sides=selected_sides, source_kind=args.source_kind,
                )
                print(json.dumps({
                    "event": "month_symbol_batch_partial", "month": f"{month:%Y-%m}",
                    "sides": sorted(pending), **progress,
                }), flush=True)
                continue
        else:
            materialised = _materialise_month_sides(pending, args.minute_root)
        for side in selected_sides:
            destination = destinations[side]
            key = (f"{month:%Y-%m}", side)
            if side in materialised:
                _atomic_parquet(materialised[side], destination)
                records_by_key[key] = _coverage_record(destination, f"{month:%Y-%m}", side, "materialised")
            record = records_by_key[key]
            _write_status(
                args.out, records_by_key.values(), source=source, minute_root=args.minute_root,
                start=start, end=end, required_months=all_months,
                required_sides=selected_sides, source_kind=args.source_kind,
            )
            print(json.dumps({"event": "month_side_complete", **record}), flush=True)
        if all(
            (args.out / "parts" / f"month={month:%Y-%m}" / f"side={side}.parquet").exists()
            for side in SIDES
        ):
            shutil.rmtree(args.out / "symbol_parts" / f"month={month:%Y-%m}", ignore_errors=True)


if __name__ == "__main__":
    main()
