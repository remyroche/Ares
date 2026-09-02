#!/usr/bin/env python3
"""Materialise exact-path opportunity and conversion labels for short P0 winners.

The script is deliberately target-only.  It starts from immutable, target-free
P0 rank-1 identities, joins their frozen exact-H12 entry/ATR fields, reopens
the canonical post-decision 720 x one-minute path, and derives realised path
labels.  Nothing written here may be used as an inference feature.

Invalid or incomplete paths remain null-labelled.  They are never converted
into ordinary economic zeroes, and every valid canonical-policy outcome is
checked against the source P0 ledger before output is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_policy_contract import (  # noqa: E402
    Exact1mExecutionContract,
    Exact1mPolicyParams,
    simulate_exact_1m_parent_policy,
)
from scripts.materialize_packb_tp6_sl4_h12_labels import (  # noqa: E402
    _minute_path_pruned,
    _packb_to_kraken_symbol,
)


SCHEMA = "strict_r3_short_p0_rich_path_labels_v1"
SIDE = "short"
HORIZON_MINUTES = 12 * 60
WINDOWS = ((15, "15m"), (30, "30m"), (60, "1h"), (120, "2h"), (180, "3h"), (360, "6h"), (720, "12h"))
CLEAR_THRESHOLDS_BPS = (100, 150, 200, 300, 400)
POLICY_COST_BPS = 100.0
CANONICAL_MEDIAN_ATR_FRACTION = 0.009867850394711882
IDENTITY = ("candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="raise")


def _finite(series: pd.Series | np.ndarray) -> np.ndarray:
    return np.isfinite(np.asarray(series, dtype=float))


def _month_key(value: pd.Series) -> pd.Series:
    return _utc(value).dt.strftime("%Y-%m")


def _part(root: Path, month: str) -> Path | None:
    path = root / "parts" / f"month={month}" / "side=short.parquet"
    return path if path.exists() else None


def _root_by_month(roots: Iterable[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for root in roots:
        for part in root.glob("parts/month=*/side=short.parquet"):
            month = part.parent.name.removeprefix("month=")
            if month in result:
                raise ValueError(f"duplicate side-local source for month {month}: {result[month]} / {root}")
            result[month] = root
    return result


def _canonical_policy() -> Exact1mPolicyParams:
    return Exact1mPolicyParams(sl_mult=3.0, trailing_activation_mult=0.50, fixed_trailing_gap_mult=0.25)


def _path_matrices(minute: pd.DataFrame, decisions: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact decision-minute open/high/low/close path arrays."""
    starts = minute.index.get_indexer(pd.DatetimeIndex(decisions)).astype(np.int64)
    offsets = np.arange(HORIZON_MINUTES, dtype=np.int64)[None, :]
    positions = starts[:, None] + offsets
    in_range = (starts >= 0) & (positions[:, -1] < len(minute))
    safe = np.clip(positions, 0, max(len(minute) - 1, 0))
    numeric = minute.loc[:, ["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    entry_open = numeric["open"].to_numpy(np.float64)[safe[:, 0]]
    high = numeric["high"].to_numpy(np.float64)[safe]
    low = numeric["low"].to_numpy(np.float64)[safe]
    close = numeric["close"].to_numpy(np.float64)[safe]
    complete = in_range & np.isfinite(entry_open) & np.isfinite(high).all(axis=1) & np.isfinite(low).all(axis=1) & np.isfinite(close).all(axis=1)
    return entry_open, high, low, close, complete


def _first_hit(values: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    hit = values >= float(threshold)
    reached = hit.any(axis=1)
    minute = np.where(reached, np.argmax(hit, axis=1).astype(float) + 1.0, np.nan)
    return reached, minute


def _max_run(mask: np.ndarray) -> np.ndarray:
    """Maximum consecutive true run, vectorised across path rows."""
    current = np.zeros(len(mask), dtype=np.int16)
    maximum = np.zeros(len(mask), dtype=np.int16)
    for column in range(mask.shape[1]):
        current = np.where(mask[:, column], current + 1, 0).astype(np.int16)
        maximum = np.maximum(maximum, current)
    return maximum


def _direction_reversals(moves: np.ndarray) -> np.ndarray:
    signs = np.sign(moves).astype(np.int8)
    result = np.zeros(len(signs), dtype=np.int16)
    for row in range(len(signs)):
        nonzero = signs[row][signs[row] != 0]
        result[row] = max(0, int(np.count_nonzero(nonzero[1:] != nonzero[:-1])))
    return result


def _path_metrics(*, entry: np.ndarray, atr: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> dict[str, np.ndarray]:
    """Build exact, side-normalised realised path labels for short entries."""
    entry = np.asarray(entry, dtype=float)
    atr = np.asarray(atr, dtype=float)
    atr_bps = np.maximum(atr / entry * 10_000.0, 1e-8)
    favourable_bps = np.maximum(0.0, 1.0 - low / entry[:, None]) * 10_000.0
    adverse_bps = np.maximum(0.0, high / entry[:, None] - 1.0) * 10_000.0
    result: dict[str, np.ndarray] = {}

    for minutes, suffix in WINDOWS:
        mfe = np.max(favourable_bps[:, :minutes], axis=1)
        result[f"mfe_{suffix}_bps"] = mfe.astype(np.float32)
        result[f"mfe_{suffix}_atr"] = (mfe / atr_bps).astype(np.float32)
        mae = np.max(adverse_bps[:, :minutes], axis=1)
        result[f"mae_{suffix}_bps"] = mae.astype(np.float32)
        result[f"mae_{suffix}_atr"] = (mae / atr_bps).astype(np.float32)

    result["max_gross_opportunity_bps"] = result["mfe_12h_bps"].copy()
    result["max_net_opportunity_bps"] = (result["mfe_12h_bps"].astype(float) - POLICY_COST_BPS).astype(np.float32)
    running_adverse = np.maximum.accumulate(adverse_bps, axis=1)
    for threshold in CLEAR_THRESHOLDS_BPS:
        reached, first = _first_hit(favourable_bps, threshold)
        index = np.maximum(np.nan_to_num(first, nan=1.0).astype(int) - 1, 0)
        before = running_adverse[np.arange(len(entry)), index]
        result[f"reached_{threshold}bps"] = reached.astype(bool)
        result[f"time_to_{threshold}bps_minutes"] = first.astype(np.float32)
        result[f"mae_before_{threshold}bps_bps"] = np.where(reached, before, np.nan).astype(np.float32)
        result[f"mae_before_{threshold}bps_atr"] = np.where(reached, before / atr_bps, np.nan).astype(np.float32)
        ratio = np.full(len(entry), np.nan, dtype=float)
        np.divide(before, result["mfe_12h_bps"], out=ratio, where=reached & (result["mfe_12h_bps"] > 0.0))
        result[f"mae_before_to_mfe12_ratio_{threshold}bps"] = ratio.astype(np.float32)

    max_mae_index = np.argmax(adverse_bps, axis=1)
    result["time_to_max_mae_minutes"] = (max_mae_index + 1).astype(np.float32)
    peak3_index = np.argmax(favourable_bps[:, :180], axis=1)
    result["mae_before_mfe_3h_bps"] = running_adverse[np.arange(len(entry)), peak3_index].astype(np.float32)
    result["mae_before_mfe_3h_atr"] = (running_adverse[np.arange(len(entry)), peak3_index] / atr_bps).astype(np.float32)
    result["time_to_mfe_3h_minutes"] = (peak3_index + 1).astype(np.float32)

    # Price changes use the frozen decision open as the first anchor, then
    # exact one-minute closes.  Negative moves are favourable for shorts.
    moves = np.concatenate([close[:, :1] - entry[:, None], np.diff(close, axis=1)], axis=1)
    previous = np.concatenate([entry[:, None], close[:, :-1]], axis=1)
    returns = np.divide(moves, previous, out=np.zeros_like(moves), where=np.abs(previous) > 1e-12)
    for minutes, suffix in ((30, "30m"), (60, "1h"), (180, "3h"), (720, "12h")):
        local_moves = moves[:, :minutes]
        denom = np.sum(np.abs(local_moves), axis=1)
        terminal = (entry - close[:, minutes - 1]) / entry
        result[f"path_efficiency_{suffix}"] = np.divide(np.abs(terminal * entry), denom, out=np.zeros_like(denom), where=denom > 1e-12).astype(np.float32)
        result[f"short_directional_efficiency_{suffix}"] = np.divide(terminal * entry, denom, out=np.zeros_like(denom), where=denom > 1e-12).astype(np.float32)
        result[f"fraction_negative_1m_bars_{suffix}"] = np.mean(local_moves < 0.0, axis=1).astype(np.float32)
        realized_vol_bps = np.sqrt(np.sum(np.square(returns[:, :minutes]), axis=1)) * 10_000.0
        result[f"realized_vol_{suffix}_bps"] = realized_vol_bps.astype(np.float32)
        result[f"downside_return_over_realized_vol_{suffix}"] = np.divide(terminal * 10_000.0, realized_vol_bps, out=np.zeros_like(realized_vol_bps), where=realized_vol_bps > 1e-12).astype(np.float32)

    signs = np.sign(moves).astype(np.int8)
    nonzero_counts = (signs != 0).sum(axis=1)
    reversals = _direction_reversals(moves)
    result["downside_run_length_12h_minutes"] = _max_run(moves < 0.0).astype(np.float32)
    result["number_direction_reversals_12h"] = reversals.astype(np.float32)
    result["drawdown_monotonicity_12h"] = np.where(
        nonzero_counts > 1,
        1.0 - reversals / np.maximum(nonzero_counts - 1, 1),
        1.0,
    ).astype(np.float32)
    return result


def _conversion_metrics(
    *, metrics: dict[str, np.ndarray], gross: np.ndarray, exit_bar: np.ndarray, exit_reason: np.ndarray,
    atr_bps: np.ndarray, mfe_before_exit_bps: np.ndarray,
) -> dict[str, np.ndarray]:
    gross = np.asarray(gross, dtype=float)
    exit_bar = np.asarray(exit_bar, dtype=int)
    reason = np.asarray(exit_reason, dtype=object).astype(str)
    opportunity = np.asarray(metrics["mfe_12h_bps"], dtype=float)
    before_exit = np.asarray(mfe_before_exit_bps, dtype=float)
    capture_positive = np.full(len(gross), np.nan, dtype=float)
    np.divide(np.maximum(gross, 0.0), opportunity, out=capture_positive, where=opportunity > 0.0)
    capture_positive = np.clip(capture_positive, 0.0, 1.0)
    capture_cost_clear = np.where(opportunity >= POLICY_COST_BPS, capture_positive, np.nan)
    activation_bps = 0.5 * atr_bps
    reached_100 = np.asarray(metrics["reached_100bps"], dtype=bool)
    time_100 = np.asarray(metrics["time_to_100bps_minutes"], dtype=float)
    early_adverse = np.asarray(metrics["mae_before_100bps_atr"], dtype=float) >= 1.0
    stop = reason == "stop_loss"
    trailing = reason == "trailing"
    timeout = reason == "timeout"
    activation_reached = opportunity >= activation_bps
    activation_failure = activation_reached & timeout
    late = reached_100 & (time_100 > 360.0)
    giveback_bps = np.maximum(before_exit - gross, 0.0)
    giveback_ratio = np.full(len(gross), np.nan, dtype=float)
    np.divide(giveback_bps, before_exit, out=giveback_ratio, where=before_exit > 0.0)
    giveback_ratio = np.clip(giveback_ratio, 0.0, 1.0)
    post_activation_giveback = trailing & activation_reached & (giveback_bps > 0.0)
    category = np.full(len(gross), "captured_or_other", dtype=object)
    category[opportunity < POLICY_COST_BPS] = "opportunity_never_existed"
    category[late] = "opportunity_arrived_too_late"
    category[early_adverse & reached_100] = "early_adverse_before_clear"
    category[activation_failure] = "trailing_activation_failure"
    category[post_activation_giveback] = "post_activation_giveback"
    category[timeout] = "timeout"
    category[stop] = "stop_out"
    return {
        "policy_exit_bar_0based": exit_bar.astype(np.float32),
        "policy_exit_minute_1based": (exit_bar + 1).astype(np.float32),
        "policy_exit_reason": reason,
        "policy_gross_bps": gross.astype(np.float32),
        "policy_net_bps": (gross - POLICY_COST_BPS).astype(np.float32),
        "policy_capture_ratio_mfe_positive": capture_positive.astype(np.float32),
        "policy_capture_ratio_cost_clear": capture_cost_clear.astype(np.float32),
        "policy_mfe_before_exit_bps": before_exit.astype(np.float32),
        "policy_giveback_bps": giveback_bps.astype(np.float32),
        "policy_giveback_ratio": giveback_ratio.astype(np.float32),
        "policy_regret_bps": (opportunity - gross).astype(np.float32),
        "policy_opportunity_never_existed": (opportunity < POLICY_COST_BPS),
        "policy_opportunity_arrived_too_late": late,
        "policy_early_adverse_before_clear": early_adverse & reached_100,
        "policy_trailing_activation_reached": activation_reached,
        "policy_trailing_activation_failure": activation_failure,
        "policy_post_activation_giveback": post_activation_giveback,
        "policy_stop_out": stop,
        "policy_timeout": timeout,
        "policy_trailing_exit": trailing,
        "policy_conversion_category": category,
    }


def _mfe_before_exit(*, favourable_bps: np.ndarray, exit_bar: np.ndarray) -> np.ndarray:
    running = np.maximum.accumulate(favourable_bps, axis=1)
    safe = np.clip(np.asarray(exit_bar, dtype=int), 0, favourable_bps.shape[1] - 1)
    return running[np.arange(len(favourable_bps)), safe]


def _join_month(*, population: pd.DataFrame, h12_root: Path, policy_root: Path | None, month: str) -> pd.DataFrame:
    h12_path = _part(h12_root, month)
    if h12_path is None:
        raise FileNotFoundError(f"no H12 label part for {month} in {h12_root}")
    h12 = pd.read_parquet(h12_path, columns=[
        *IDENTITY, "__label_available_at__", "tp6_sl4_entry_price", "atr_1h", "label_valid", "target_invalid",
    ])
    policy: pd.DataFrame | None = None
    if policy_root is not None and (policy_path := _part(policy_root, month)) is not None:
        policy = pd.read_parquet(policy_path, columns=[
            "candidate_id", "policy_path_valid", "policy_label_available_at", "p0_canonical_gross_bps", "p0_canonical_net_bps",
            "p0_canonical_exit_minute", "p0_canonical_exit_reason",
        ])
        if policy.candidate_id.duplicated().any():
            raise ValueError(f"duplicate policy identities for {month}")
    if h12.candidate_id.duplicated().any():
        raise ValueError(f"duplicate H12 identities for {month}")
    wanted = population.loc[:, [*IDENTITY, "p0_canonical_net_bps", "policy_path_valid", "policy_label_available_at"]].copy()
    wanted = wanted.merge(h12, on=list(IDENTITY), how="left", validate="one_to_one", suffixes=("", "_h12"))
    if wanted["tp6_sl4_entry_price"].isna().all():
        raise ValueError(f"P0 population has no overlapping H12 identities for {month}")
    if policy is not None:
        wanted = wanted.merge(policy, on="candidate_id", how="left", validate="one_to_one", suffixes=("_population", "_policy"))
        # The population source is authoritative for M4 lineage; the existing
        # policy ledger is a second exact-policy parity receipt.
        source = pd.to_numeric(wanted["p0_canonical_net_bps_population"], errors="coerce")
        receipt = pd.to_numeric(wanted["p0_canonical_net_bps_policy"], errors="coerce")
        both = source.notna() & receipt.notna()
        if both.any() and not np.allclose(source[both], receipt[both], rtol=0.0, atol=2e-4):
            raise AssertionError(f"P0 population/policy receipt mismatch in {month}")
        population_available = _utc(wanted["policy_label_available_at_population"])
        policy_available = _utc(wanted["policy_label_available_at_policy"])
        shared_available = population_available.notna() & policy_available.notna()
        if shared_available.any() and not population_available[shared_available].eq(policy_available[shared_available]).all():
            raise AssertionError(f"P0 population/policy availability mismatch in {month}")
        wanted["policy_label_available_at"] = policy_available.where(policy_available.notna(), population_available)
        wanted["source_policy_net_bps"] = source
        wanted["source_policy_path_valid"] = wanted["policy_path_valid_policy"].fillna(False).astype(bool)
    else:
        wanted["source_policy_net_bps"] = pd.to_numeric(wanted["p0_canonical_net_bps"], errors="coerce")
        wanted["source_policy_path_valid"] = wanted["policy_path_valid"].fillna(False).astype(bool)
    for column in ("__ts__", "__decision_ts__", "__label_available_at__", "policy_label_available_at"):
        wanted[column] = _utc(wanted[column])
    if not wanted.side_name.astype(str).str.lower().eq(SIDE).all():
        raise ValueError("P0 rich labels are short-only")
    if not wanted["__decision_ts__"].eq(wanted["__ts__"] + pd.Timedelta(hours=1)).all():
        raise AssertionError("P0 labels require signal close + one-hour decision entry")
    if not wanted["__label_available_at__"].eq(wanted["__decision_ts__"] + pd.Timedelta(hours=12)).all():
        raise AssertionError("H12 availability must equal decision + 12 hours")
    return wanted.sort_values(["__symbol__", "__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _blank_labels(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    output = source.loc[:, [*IDENTITY, "__label_available_at__"]].copy()
    output["rich_path_label_valid"] = False
    output["rich_path_target_invalid"] = True
    numeric = {
        name: np.full(len(source), np.nan, dtype=np.float32)
        for name in (
            *[f"mfe_{suffix}_{unit}" for _minutes, suffix in WINDOWS for unit in ("bps", "atr")],
            *[f"mae_{suffix}_{unit}" for _minutes, suffix in WINDOWS for unit in ("bps", "atr")],
            "max_gross_opportunity_bps", "max_net_opportunity_bps", "time_to_max_mae_minutes",
            "mae_before_mfe_3h_bps", "mae_before_mfe_3h_atr", "time_to_mfe_3h_minutes",
            *[f"time_to_{threshold}bps_minutes" for threshold in CLEAR_THRESHOLDS_BPS],
            *[f"mae_before_{threshold}bps_{unit}" for threshold in CLEAR_THRESHOLDS_BPS for unit in ("bps", "atr")],
            *[f"mae_before_to_mfe12_ratio_{threshold}bps" for threshold in CLEAR_THRESHOLDS_BPS],
            *[f"path_efficiency_{suffix}" for suffix in ("30m", "1h", "3h", "12h")],
            *[f"short_directional_efficiency_{suffix}" for suffix in ("30m", "1h", "3h", "12h")],
            *[f"fraction_negative_1m_bars_{suffix}" for suffix in ("30m", "1h", "3h", "12h")],
            *[f"realized_vol_{suffix}_bps" for suffix in ("30m", "1h", "3h", "12h")],
            *[f"downside_return_over_realized_vol_{suffix}" for suffix in ("30m", "1h", "3h", "12h")],
            "downside_run_length_12h_minutes", "number_direction_reversals_12h", "drawdown_monotonicity_12h",
            "policy_exit_bar_0based", "policy_exit_minute_1based", "policy_gross_bps", "policy_net_bps",
            "policy_capture_ratio_mfe_positive", "policy_capture_ratio_cost_clear", "policy_mfe_before_exit_bps",
            "policy_giveback_bps", "policy_giveback_ratio", "policy_regret_bps",
        )
    }
    flags = {
        name: np.full(len(source), None, dtype=object)
        for name in (
            *[f"reached_{threshold}bps" for threshold in CLEAR_THRESHOLDS_BPS],
            "policy_opportunity_never_existed", "policy_opportunity_arrived_too_late", "policy_early_adverse_before_clear",
            "policy_trailing_activation_reached", "policy_trailing_activation_failure", "policy_post_activation_giveback",
            "policy_stop_out", "policy_timeout", "policy_trailing_exit",
        )
    }
    return output, numeric, flags


def _materialize_month(*, source: pd.DataFrame, minute_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    output, numeric, flags = _blank_labels(source)
    categories = np.full(len(source), None, dtype=object)
    reasons = np.full(len(source), None, dtype=object)
    h12_valid = source["label_valid"].fillna(False).astype(bool) & ~source["target_invalid"].fillna(True).astype(bool)
    valid_index = np.flatnonzero(h12_valid.to_numpy())
    simulated_parity_rows = 0
    source_parity_rows = 0
    for symbol, group in source.loc[h12_valid].groupby("__symbol__", sort=True):
        minute = _minute_path_pruned(
            minute_root,
            _packb_to_kraken_symbol(str(symbol)),
            group["__decision_ts__"].min(),
            group["__decision_ts__"].max() + pd.Timedelta(minutes=HORIZON_MINUTES),
        )
        path_open, high, low, close, complete = _path_matrices(minute, group["__decision_ts__"])
        if not complete.all():
            failures = group.loc[~complete, "candidate_id"].astype(str).head(5).tolist()
            raise AssertionError(f"valid short P0 winner lacks complete exact H12 path: {symbol}: {failures}")
        rows = group.index.to_numpy(dtype=int)
        entry = pd.to_numeric(group["tp6_sl4_entry_price"], errors="coerce").to_numpy(float)
        atr = pd.to_numeric(group["atr_1h"], errors="coerce").to_numpy(float)
        if not np.isfinite(entry).all() or not np.isfinite(atr).all() or (entry <= 0.0).any() or (atr <= 0.0).any():
            raise AssertionError(f"valid H12 P0 winner lacks frozen entry/ATR: {symbol}")
        if not np.allclose(path_open, entry, rtol=0.0, atol=1e-12):
            raise AssertionError(f"reopened decision-minute open differs from frozen short entry: {symbol}")
        policy = simulate_exact_1m_parent_policy(
            entry=entry, atr=atr, highs=high, lows=low, closes=close,
            entry_timestamps=pd.DatetimeIndex(group["__decision_ts__"]), params=_canonical_policy(),
            contract=Exact1mExecutionContract(entry_delay_minutes=0),
            median_atr_fraction=CANONICAL_MEDIAN_ATR_FRACTION, side=SIDE,
        )
        path_valid = np.asarray(policy["path_valid"], dtype=bool)
        if not path_valid.all():
            raise AssertionError(f"canonical policy invalidated complete H12 path: {symbol}")
        policy_net = np.asarray(policy["net_bps"], dtype=float)
        source_net = pd.to_numeric(group["source_policy_net_bps"], errors="coerce").to_numpy(float)
        comparable = np.isfinite(source_net)
        if comparable.any() and not np.allclose(policy_net[comparable], source_net[comparable], rtol=0.0, atol=2e-4):
            mismatch = group.iloc[np.flatnonzero(comparable & ~np.isclose(policy_net, source_net, rtol=0.0, atol=2e-4))]["candidate_id"].astype(str).head(5).tolist()
            raise AssertionError(f"canonical policy reproduction differs from source P0 net: {symbol}: {mismatch}")
        source_parity_rows += int(comparable.sum())
        simulated_parity_rows += len(group)
        metrics = _path_metrics(entry=entry, atr=atr, high=high, low=low, close=close)
        favourable = np.maximum(0.0, 1.0 - low / entry[:, None]) * 10_000.0
        exit_bar = np.asarray(policy["exit_bar"], dtype=int)
        metrics["mfe_12h_bps"] = np.asarray(metrics["mfe_12h_bps"], dtype=np.float32)
        mfe_before_exit = _mfe_before_exit(favourable_bps=favourable, exit_bar=exit_bar)
        conversion = _conversion_metrics(
            metrics=metrics, gross=np.asarray(policy["gross_bps"], dtype=float), exit_bar=exit_bar,
            exit_reason=np.asarray(policy["exit_reason"], dtype=object), atr_bps=atr / entry * 10_000.0,
            mfe_before_exit_bps=mfe_before_exit,
        )
        for name, values in metrics.items():
            if name in numeric:
                numeric[name][rows] = np.asarray(values, dtype=np.float32)
            elif name in flags:
                flags[name][rows] = np.asarray(values, dtype=bool)
            else:
                raise AssertionError(f"unregistered rich-path label field: {name}")
        for name, values in conversion.items():
            if name in numeric:
                numeric[name][rows] = np.asarray(values, dtype=np.float32)
            elif name in flags:
                flags[name][rows] = np.asarray(values, dtype=bool)
            elif name == "policy_conversion_category":
                categories[rows] = np.asarray(values, dtype=object)
            elif name == "policy_exit_reason":
                reasons[rows] = np.asarray(values, dtype=object)
            else:
                raise AssertionError(f"unregistered conversion label field: {name}")
    label_columns = pd.DataFrame({
        **numeric,
        **flags,
        "policy_exit_reason": reasons,
        "policy_conversion_category": categories,
    }, index=output.index)
    output = pd.concat([output, label_columns], axis=1, copy=False)
    output.loc[valid_index, "rich_path_label_valid"] = True
    output.loc[valid_index, "rich_path_target_invalid"] = False
    invalid = ~h12_valid.to_numpy()
    supervised = [*numeric, *flags, "policy_exit_reason", "policy_conversion_category"]
    if output.loc[invalid, supervised].notna().any().any():
        raise AssertionError("invalid P0 winners received ordinary path/conversion labels")
    if not output.loc[valid_index, "rich_path_label_valid"].all():
        raise AssertionError("valid P0 winners lost rich-path label validity")
    return output, {
        "rows": int(len(output)),
        "valid_rows": int(len(valid_index)),
        "invalid_rows": int(invalid.sum()),
        "reopened_entry_parity_rows": int(simulated_parity_rows),
        "source_policy_parity_rows": int(source_parity_rows),
        "ever_100bps_rows": int(output.loc[valid_index, "reached_100bps"].sum()),
        "mean_mfe_12h_bps": float(output.loc[valid_index, "mfe_12h_bps"].mean()) if len(valid_index) else float("nan"),
        "mean_policy_net_bps": float(output.loc[valid_index, "policy_net_bps"].mean()) if len(valid_index) else float("nan"),
    }


@dataclass(frozen=True)
class Sources:
    population_roots: tuple[Path, ...]
    h12_roots: tuple[Path, ...]
    policy_roots: tuple[Path, ...]


def _load_population(roots: Iterable[Path], *, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, int]]:
    pieces: list[pd.DataFrame] = []
    for source_order, root in enumerate(roots):
        path = root / "short_p0_top1_hourly_population.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        columns = [*IDENTITY, "p0_canonical_net_bps", "policy_path_valid", "policy_label_available_at"]
        frame = pd.read_parquet(path, columns=columns)
        frame["__decision_ts__"] = _utc(frame["__decision_ts__"])
        frame = frame.loc[frame["__decision_ts__"].ge(start) & frame["__decision_ts__"].lt(end)].copy()
        frame["__source_order__"] = source_order
        pieces.append(frame)
    result = pd.concat(pieces, ignore_index=True)
    for column in ("__ts__", "__decision_ts__", "policy_label_available_at"):
        result[column] = _utc(result[column])
    if not result.side_name.astype(str).str.lower().eq(SIDE).all():
        raise ValueError("P0 source is not short-only")
    duplicate_rows = int(result.candidate_id.duplicated(keep=False).sum())
    duplicate_ids = int(result.loc[result.candidate_id.duplicated(keep=False), "candidate_id"].nunique())
    if duplicate_rows:
        # The absolute-conversion roots are cumulative immutable snapshots.
        # A later snapshot may supersede an earlier one only after every shared
        # candidate identity and source outcome is exactly identical.
        canonical = result.sort_values("__source_order__", kind="stable").drop_duplicates("candidate_id", keep="last").set_index("candidate_id")
        comparison_fields = ("__ts__", "__decision_ts__", "__symbol__", "side_name", "p0_canonical_net_bps", "policy_path_valid", "policy_label_available_at")
        repeated = result.loc[result.candidate_id.duplicated(keep=False), ["candidate_id", *comparison_fields]].copy()
        for field in comparison_fields:
            left = repeated[field]
            right = repeated["candidate_id"].map(canonical[field])
            if pd.api.types.is_numeric_dtype(left) and not pd.api.types.is_bool_dtype(left):
                equal = np.isclose(left.to_numpy(float), pd.to_numeric(right, errors="coerce").to_numpy(float), rtol=0.0, atol=2e-4, equal_nan=True)
            else:
                equal = (left.eq(right) | (left.isna() & right.isna())).fillna(False).to_numpy(bool)
            if not equal.all():
                raise ValueError(f"overlapping P0 source artifacts disagree on {field}")
        result = canonical.reset_index()
    if result.candidate_id.duplicated().any():
        raise ValueError("P0 source identity deduplication failed")
    if not result["__decision_ts__"].eq(result["__ts__"] + pd.Timedelta(hours=1)).all():
        raise AssertionError("P0 candidate entry convention changed")
    result = result.drop(columns="__source_order__", errors="ignore")
    return result.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True), {
        "input_rows": int(sum(len(piece) for piece in pieces)),
        "unique_rows": int(len(result)),
        "overlapping_source_rows": duplicate_rows,
        "overlapping_source_candidate_ids": duplicate_ids,
    }


def run(*, sources: Sources, minute_root: Path, out: Path, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    population, population_audit = _load_population(sources.population_roots, start=start, end=end)
    h12_by_month = _root_by_month(sources.h12_roots)
    policy_by_month = _root_by_month(sources.policy_roots)
    records: list[dict[str, Any]] = []
    for month, block in population.groupby(_month_key(population["__decision_ts__"]), sort=True):
        if month not in h12_by_month:
            raise FileNotFoundError(f"no exact H12 source covers P0 winner month {month}")
        joined = _join_month(
            population=block.reset_index(drop=True), h12_root=h12_by_month[month],
            policy_root=policy_by_month.get(month), month=month,
        )
        labels, audit = _materialize_month(source=joined, minute_root=minute_root)
        if len(labels) != len(block) or labels.candidate_id.duplicated().any():
            raise AssertionError(f"P0 rich path labels changed identities for {month}")
        destination = out / "parts" / f"month={month}" / "side=short.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(destination, index=False, compression="zstd")
        audit["month"] = month
        records.append(audit)
        print(json.dumps(audit, sort_keys=True), flush=True)
    coverage = pd.DataFrame(records)
    coverage.to_parquet(out / "coverage_by_month.parquet", index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "side": SIDE,
        "decision_window": f"[{start.isoformat()}, {end.isoformat()})",
        "population": "frozen P0 rank-1 hourly population only; source contains target-free identities before outcomes are joined",
        "entry": "frozen exact decision-minute open at signal close + one hour; rechecked against reopened 1m OHLCV",
        "horizon": "complete post-decision 720 x one-minute bars",
        "label_available_at": "decision + 12 hours",
        "policy": "canonical short parent policy: SL 3 ATR, trailing activation 0.5 ATR, giveback 0.25 ATR, H12 timeout, cost 100 bps once",
        "policy_median_atr_fraction": CANONICAL_MEDIAN_ATR_FRACTION,
        "path_definitions": {
            "favourable_short_move": "entry - low",
            "adverse_short_move": "high - entry",
            "path_efficiency": "abs(entry - close_h) / sum(abs(exact one-minute close changes including entry-to-first-close))",
            "drawdown_monotonicity": "one minus the exact-one-minute nonzero-sign reversal rate",
            "capture": "clip(max(policy gross, 0) / MFE, 0, 1)",
            "regret": "MFE_12h - policy gross",
        },
        "invalidity": "invalid/incomplete H12 rows are null-labelled and excluded from supervised use; never encoded as economic failures",
        "inference": "every path/conversion field is supervised-label-only and forbidden from inference feature contracts",
        "sources": {
            "population_roots": [str(path.resolve()) for path in sources.population_roots],
            "h12_roots": [str(path.resolve()) for path in sources.h12_roots],
            "policy_roots": [str(path.resolve()) for path in sources.policy_roots],
            "minute_root": str(minute_root.resolve()),
            "manifests_sha256": {
                str(path.resolve()): _sha256(path / "run_manifest.json")
                for path in (*sources.population_roots, *sources.h12_roots, *sources.policy_roots)
                if (path / "run_manifest.json").exists()
            },
        },
        "coverage": records,
        "population_source_audit": population_audit,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", action="append", type=Path, required=True)
    parser.add_argument("--h12-root", action="append", type=Path, required=True)
    parser.add_argument("--policy-root", action="append", type=Path, default=[])
    parser.add_argument("--minute-root", type=Path, default=ROOT / "data_perp/exchanges/krakenfutures/execution_1m/ohlcv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="2024-05-01T00:00:00Z")
    parser.add_argument("--end", default="2026-08-01T00:00:00Z")
    args = parser.parse_args()
    print(run(
        sources=Sources(tuple(path.resolve() for path in args.population_root), tuple(path.resolve() for path in args.h12_root), tuple(path.resolve() for path in args.policy_root)),
        minute_root=args.minute_root.resolve(), out=args.out.resolve(), start=pd.Timestamp(args.start), end=pd.Timestamp(args.end),
    ))


if __name__ == "__main__":
    main()
