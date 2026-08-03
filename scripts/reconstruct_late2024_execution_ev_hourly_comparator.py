#!/usr/bin/env python3
"""Build a strict two-layer Oct--Dec 2024 *hourly* execution-EV comparator.

This is deliberately a separate evidence tier.  It starts from the rows that
exist in the point-in-time feature store at every timestamp (rather than a
later archived score/candidate list), creates one long and one short candidate
per available asset, and simulates the current side-parent policy on canonical
hourly bars.  The July--September history is warm-up/training only; reported
two-layer OOF scores are October--December only.

It must never be described as exact-1m exit, entry-timing, or L2/spread
evidence, and its metrics must not be pooled with the exact-1m reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.types as pat
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import MODEL_DIRECT_BASE_FEATURE_KEYS  # noqa: E402
from extreme_price_movements.data_store import PartitionedOHLCVStore  # noqa: E402
from extreme_price_movements.execution_ev_labels import (  # noqa: E402
    reason_names,
    simulate_execution_ev_12h,
)
from scripts.backfill_historical_execution_ev_12h_oof import _current_geometry  # noqa: E402


SCHEMA = "late2024_execution_ev_hourly_comparator_two_layer_oof_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
TARGET = "execution_net_ev_12h"
BASE_TARGET = "execution_soft_positive_12h"
BASE_SCORE = "hourly_base_soft_oof"
DIRECT_SCORE = "hourly_execution_ev_oof"
HORIZON_HOURS = 12
DECISION_DELAY_HOURS = 1
BASE_WARMUP_DAYS = 14
FOLD_DAYS = 7
MIN_BASE_TRAIN_ROWS = 20_000
MIN_META_TRAIN_ROWS = 20_000
MAX_SELECTED_FEATURES = 32
MAX_FIT_ROWS = 150_000
SOFT_LABEL_SCALE_RETURN = 0.01
SEED = 202410
SIDES = ("long", "short")
META_FEATURES = (
    BASE_SCORE,
    "base_margin_to_cutoff",
    "base_margin_to_cutoff_z",
    "base_score_z_within_timestamp",
    "base_score_rank_pct_within_timestamp",
    "candidate_group_size",
)
FORBIDDEN_RAW_TOKENS = (
    "future",
    "label",
    "target",
    "outcome",
    "execution_",
    "_mfe",
    "_mae",
    "_pnl",
    "fund",
)


def _safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    clean = {str(key): _safe(value) for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _utc(values: pd.Series | Iterable[Any], *, name: str) -> pd.Series:
    result = pd.Series(pd.to_datetime(values, utc=True, errors="coerce"))
    if result.isna().any():
        raise ValueError(f"{name} contains invalid UTC timestamps")
    return result


def normalize_symbol(value: Any) -> str:
    symbol = str(value).strip().replace("/", "_")
    if not symbol:
        raise ValueError("blank symbol")
    return symbol


def stable_candidate_id(timestamp: pd.Timestamp, symbol: str, side: str) -> str:
    payload = f"{SCHEMA}|{pd.Timestamp(timestamp).isoformat()}|{symbol}|{side}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _feature_file_symbol(path: Path) -> str:
    stem = path.name
    if not stem.startswith("symbol=") or not stem.endswith(".parquet"):
        raise ValueError(f"feature path is not a symbol parquet: {path}")
    return normalize_symbol(stem[len("symbol=") : -len(".parquet")])


def _timestamp_column(names: Iterable[str]) -> str | None:
    known = set(names)
    if "ts" in known:
        return "ts"
    if "__index_level_0__" in known:
        return "__index_level_0__"
    return None


def eligible_raw_features(
    paths: Iterable[Path],
    *,
    configured_features: Sequence[str] = MODEL_DIRECT_BASE_FEATURE_KEYS,
    minimum_features: int = 12,
) -> tuple[list[str], dict[str, Any]]:
    """Return a predeclared raw/PIT pool, never target-selected globally.

    The source schemas are heterogeneous.  A selected field may be absent for
    a particular asset and is then an explicit missing value, imputed inside
    each fold.  Requiring a global intersection would introduce survivorship by
    dropping assets solely because a feature was added later.  The historical
    funding source is not authoritative before 2026, so every funding-family
    field is excluded even if a cache happens to contain a value.
    """

    numeric_counts: dict[str, int] = {str(name): 0 for name in configured_features}
    timestamp_columns: dict[str, int] = {"ts": 0, "__index_level_0__": 0, "missing": 0}
    files = list(paths)
    for path in files:
        schema = pq.ParquetFile(path).schema_arrow
        names = {field.name: field for field in schema}
        clock = _timestamp_column(names)
        timestamp_columns[clock or "missing"] += 1
        for name in numeric_counts:
            field = names.get(name)
            if field is not None and (pat.is_floating(field.type) or pat.is_integer(field.type) or pat.is_boolean(field.type)):
                numeric_counts[name] += 1
    selected = [
        name
        for name in configured_features
        if numeric_counts.get(name, 0) > 0
        and not any(token in name.lower() for token in FORBIDDEN_RAW_TOKENS)
    ]
    if len(selected) < minimum_features:
        raise ValueError(f"too few eligible raw PIT feature fields: {len(selected)} < {minimum_features}")
    return selected, {
        "feature_files": len(files),
        "timestamp_physical_columns": timestamp_columns,
        "configured_raw_pool_size": len(configured_features),
        "eligible_raw_pool_size": len(selected),
        "feature_file_presence": numeric_counts,
    }


def _read_feature_file(
    path: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    features: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    schema = pq.ParquetFile(path).schema_arrow
    names = set(schema.names)
    clock = _timestamp_column(names)
    if clock is None:
        raise ValueError(f"{path} has no physical UTC timestamp column")
    available = [name for name in features if name in names]
    columns = [clock, *available]
    has_row_symbol = "__symbol__" in names
    if has_row_symbol:
        columns.append("__symbol__")
    # This predicate is a resource bound as well as a causality guard: the
    # static store spans 2022--2026, whereas this comparator may touch only
    # July--December 2024.  Do not silently read the entire future store and
    # then filter it in memory.
    frame = pd.read_parquet(
        path,
        columns=list(dict.fromkeys(columns)),
        filters=[(clock, ">=", start), (clock, "<", end)],
    )
    if clock in frame.columns:
        timestamp = _utc(frame.pop(clock), name=f"{path}:{clock}")
    elif frame.index.name == clock or (clock == "__index_level_0__" and isinstance(frame.index, pd.DatetimeIndex)):
        timestamp = _utc(frame.index, name=f"{path}:{clock}")
    else:
        raise ValueError(f"{path}: requested timestamp did not materialize")
    frame["__ts__"] = timestamp.to_numpy()
    frame = frame.loc[(frame["__ts__"] >= start) & (frame["__ts__"] < end)].copy()
    file_symbol = _feature_file_symbol(path)
    if has_row_symbol:
        row_symbols = frame.pop("__symbol__").map(normalize_symbol)
        if not row_symbols.eq(file_symbol).all():
            raise ValueError(f"{path}: __symbol__ disagrees with immutable file partition")
        frame["__symbol__"] = row_symbols.to_numpy()
    else:
        frame["__symbol__"] = file_symbol
    for feature in features:
        if feature not in frame:
            frame[feature] = np.float32(np.nan)
        else:
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce").astype(np.float32)
    return frame.loc[:, ["__ts__", "__symbol__", *features]], {
        "path": str(path),
        "timestamp_column": clock,
        "available_feature_count": len(available),
        "rows_in_window": int(len(frame)),
    }


def load_pit_candidate_universe(
    features_root: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    configured_features: Sequence[str] = MODEL_DIRECT_BASE_FEATURE_KEYS,
    minimum_features: int = 12,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Load only feature-store rows observed in ``[start, end)``.

    This is the complete point-in-time candidate universe.  It does not use an
    all-period listing screen, frozen score, future candidate status, or a
    hindsight liquidity filter.
    """

    paths = sorted(features_root.glob("symbol=*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no symbol parquet files under {features_root}")
    features, schema_report = eligible_raw_features(paths, configured_features=configured_features, minimum_features=minimum_features)
    parts: list[pd.DataFrame] = []
    file_reports: list[dict[str, Any]] = []
    for number, path in enumerate(paths, start=1):
        part, report = _read_feature_file(path, start=start, end=end, features=features)
        file_reports.append(report)
        if not part.empty:
            parts.append(part)
        if number == 1 or number % 50 == 0 or number == len(paths):
            print(f"[late2024-hourly] PIT {number}/{len(paths)} {path.name} rows={report['rows_in_window']}", flush=True)
    if not parts:
        raise ValueError("no point-in-time feature rows in requested interval")
    universe = pd.concat(parts, ignore_index=True)
    universe = universe.sort_values(["__ts__", "__symbol__"], kind="stable").reset_index(drop=True)
    if universe.duplicated(["__ts__", "__symbol__"], keep=False).any():
        raise ValueError("PIT universe contains duplicate timestamp/symbol rows")
    if universe["__ts__"].min() < start or universe["__ts__"].max() >= end:
        raise ValueError("PIT source filter escaped requested time interval")
    schema_report["files_read"] = file_reports
    schema_report["rows_in_requested_interval"] = int(len(universe))
    schema_report["symbols_in_requested_interval"] = int(universe["__symbol__"].nunique())
    return universe, features, schema_report


def _hourly_paths_at_signal(
    store: PartitionedOHLCVStore, symbol: str, signals: pd.Series
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Return next-bar paths and ATR observed at the *signal* bar close."""

    decisions = pd.DatetimeIndex(signals) + pd.Timedelta(hours=DECISION_DELAY_HOURS)
    start = pd.Timestamp(signals.min()) - pd.Timedelta(hours=16)
    end = pd.Timestamp(decisions.max()) + pd.Timedelta(hours=HORIZON_HOURS)
    bars = store.load(symbol, columns=["open", "high", "low", "close"], start_ts=start, end_ts=end)
    shape = (len(signals), HORIZON_HOURS)
    blank = tuple(np.full(shape, np.nan, dtype=np.float32) for _ in range(4))
    if bars.empty:
        missing = np.zeros(len(signals), dtype=bool)
        return blank, missing, missing.copy(), np.full(len(signals), np.nan, dtype=np.float32)
    bars = bars.loc[~bars.index.duplicated(keep="last")].sort_index()
    bars.index = pd.to_datetime(bars.index, utc=True)
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")
    close = pd.to_numeric(bars["close"], errors="coerce")
    prior_close = close.shift(1)
    true_range = pd.concat([(high - low), (high - prior_close).abs(), (low - prior_close).abs()], axis=1).max(axis=1)
    atr_fraction = true_range.rolling(14, min_periods=14).mean() / close
    atr = atr_fraction.reindex(pd.DatetimeIndex(signals)).to_numpy(np.float32)
    index_ns = bars.index.astype("int64").to_numpy(np.int64)
    decision_ns = decisions.astype("int64").to_numpy(np.int64)
    starts = np.searchsorted(index_ns, decision_ns)
    offsets = np.arange(HORIZON_HOURS, dtype=np.int64)
    positions = starts[:, None] + offsets[None, :]
    valid = positions[:, -1] < len(index_ns)
    local = np.flatnonzero(valid)
    if len(local):
        expected = decision_ns[local, None] + offsets[None, :] * 3_600_000_000_000
        valid[local] = np.all(index_ns[positions[local]] == expected, axis=1)
    arrays: list[np.ndarray] = []
    for column in ("open", "high", "low", "close"):
        values = pd.to_numeric(bars[column], errors="coerce").to_numpy(np.float32)
        result = np.full(shape, np.nan, dtype=np.float32)
        local = np.flatnonzero(valid)
        if len(local):
            result[local] = values[positions[local]]
        arrays.append(result)
    path_valid = valid & np.logical_and.reduce(
        [np.isfinite(item).all(axis=1) & (item > 0.0).all(axis=1) for item in arrays]
    )
    label_valid = path_valid & np.isfinite(atr) & (atr > 0.0)
    return tuple(arrays), label_valid, path_valid, atr


def materialize_hourly_labels(
    universe: pd.DataFrame,
    *,
    hourly_root: Path,
    policy: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply current geometry to next-hour 12-bar paths, fee exactly once."""

    long_geometry, short_geometry, fee = _current_geometry(policy)
    store = PartitionedOHLCVStore(str(hourly_root), timeframe="1h")
    labels: list[pd.DataFrame] = []
    coverage: list[pd.DataFrame] = []
    raw_features = [column for column in universe.columns if column not in {"__ts__", "__symbol__"}]
    groups = universe.groupby("__symbol__", sort=True).groups
    for number, (symbol, indices) in enumerate(groups.items(), start=1):
        rows = universe.loc[list(indices)].copy().reset_index(drop=True)
        paths, valid, path_valid, atr = _hourly_paths_at_signal(
            store, str(symbol), rows["__ts__"]
        )
        local_coverage = rows.loc[:, ["__ts__", "__symbol__"]].copy()
        local_coverage["complete_hourly_path"] = path_valid
        local_coverage["complete_causal_atr"] = np.isfinite(atr) & (atr > 0.0)
        local_coverage["complete_hourly_label"] = valid
        local_coverage["candidate_month"] = local_coverage["__ts__"].dt.strftime("%Y-%m")
        coverage.append(local_coverage)
        if not valid.any():
            continue
        valid_rows = rows.loc[valid].copy().reset_index(drop=True)
        arrays = tuple(item[valid] for item in paths)
        for side, sign in (("long", 1.0), ("short", -1.0)):
            local = valid_rows.copy()
            gross, net, reason, exit_bar, mfe, mae = simulate_execution_ev_12h(
                *arrays,
                np.full(len(local), sign, dtype=np.float64),
                atr[valid].astype(np.float64),
                np.full(len(local), fee, dtype=np.float64),
                long_geometry.vector(),
                short_geometry.vector(),
                60,
            )
            local["side_name"] = side
            local["candidate_id"] = [stable_candidate_id(timestamp, symbol, side) for timestamp in local["__ts__"]]
            local["execution_decision_utc"] = local["__ts__"] + pd.Timedelta(hours=DECISION_DELAY_HOURS)
            local["execution_label_end_utc"] = local["execution_decision_utc"] + pd.Timedelta(hours=HORIZON_HOURS)
            local["candidate_month"] = local["__ts__"].dt.strftime("%Y-%m")
            local["atr_fraction_14h_at_signal"] = atr[valid].astype(np.float32)
            # Preserve float64 for economic accounting. Some early illiquid
            # hourly paths have large gross moves for which float32 cannot
            # retain a one-percent gross-to-net difference exactly.
            local["execution_gross_ev_12h"] = gross.astype(np.float64)
            local[TARGET] = net.astype(np.float64)
            local["execution_cost_return"] = np.float64(fee)
            local["execution_exit_reason"] = reason_names(reason)
            local["execution_exit_hour"] = (exit_bar + 1).astype(np.int16)
            local["execution_mfe_return_12h"] = mfe.astype(np.float32)
            local["execution_mae_return_12h"] = mae.astype(np.float32)
            scaled = np.clip(local[TARGET].to_numpy(np.float64) / SOFT_LABEL_SCALE_RETURN, -30.0, 30.0)
            local[BASE_TARGET] = (1.0 / (1.0 + np.exp(-scaled))).astype(np.float32)
            labels.append(local.loc[:, [*IDENTITY, "execution_decision_utc", "execution_label_end_utc", "candidate_month", *raw_features, "atr_fraction_14h_at_signal", "execution_gross_ev_12h", TARGET, "execution_cost_return", "execution_exit_reason", "execution_exit_hour", "execution_mfe_return_12h", "execution_mae_return_12h", BASE_TARGET]])
        if number == 1 or number % 25 == 0 or number == len(groups):
            print(f"[late2024-hourly] labels {number}/{len(groups)} {symbol} complete={int(valid.sum())}/{len(rows)}", flush=True)
    if not labels:
        raise ValueError("no complete canonical hourly execution paths")
    result = pd.concat(labels, ignore_index=True).sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    coverage_result = pd.concat(coverage, ignore_index=True)
    if result.duplicated(list(IDENTITY), keep=False).any():
        raise ValueError("hourly labels contain duplicate identities")
    if not np.allclose(result["execution_gross_ev_12h"] - result["execution_cost_return"], result[TARGET], rtol=0.0, atol=1e-10):
        raise ValueError("gross-cost reconciliation failed")
    return result, coverage_result, {
        "round_trip_fee_return": fee,
        "long": long_geometry.vector().tolist(),
        "short": short_geometry.vector().tolist(),
        "atr": "14 completed hourly true-range bars ending at signal timestamp",
    }


def deterministic_fit_sample(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.copy()
    positions = np.linspace(0, len(frame) - 1, maximum, dtype=np.int64)
    return frame.iloc[positions].copy()


def select_features(train: pd.DataFrame, features: Sequence[str], *, maximum: int = MAX_SELECTED_FEATURES) -> tuple[list[str], dict[str, float]]:
    sample = deterministic_fit_sample(train.sort_values(["__ts__", "__symbol__"], kind="stable"), MAX_FIT_ROWS)
    target = pd.to_numeric(sample[BASE_TARGET], errors="coerce")
    scores: dict[str, float] = {}
    for feature in features:
        values = pd.to_numeric(sample[feature], errors="coerce")
        valid = values.notna() & target.notna()
        if int(valid.sum()) < 100 or values.loc[valid].nunique(dropna=True) < 2:
            scores[str(feature)] = 0.0
            continue
        value = values.loc[valid].corr(target.loc[valid], method="spearman")
        scores[str(feature)] = abs(float(value)) if np.isfinite(value) else 0.0
    selected = sorted(features, key=lambda name: (-scores[str(name)], str(name)))[:maximum]
    if not selected or max(scores.values(), default=0.0) <= 0.0:
        raise ValueError("fold-local raw/PIT feature selection found no usable inputs")
    return list(selected), scores


def _fit_matrix(train: pd.DataFrame, evaluation: pd.DataFrame, features: Sequence[str]) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    medians = train.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
    x_train = train.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(medians).to_numpy(np.float32)
    x_eval = evaluation.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(medians).to_numpy(np.float32)
    return x_train, x_eval, {str(key): float(value) for key, value in medians.items()}


def _model(*, direct: bool) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="squared_error", max_iter=100 if direct else 80, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1e-3, random_state=SEED + int(direct))


def weekly_boundaries(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    boundaries: list[pd.Timestamp] = []
    current = start
    while current < end:
        boundaries.append(current)
        current += pd.Timedelta(days=FOLD_DAYS)
    return boundaries


def add_candidate_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    grouped = result.groupby(["__ts__", "side_name"], sort=False)[BASE_SCORE]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    cutoff = grouped.transform(lambda values: values.quantile(0.70))
    result["base_margin_to_cutoff"] = result[BASE_SCORE] - cutoff
    result["base_margin_to_cutoff_z"] = result["base_margin_to_cutoff"] / std
    result["base_score_z_within_timestamp"] = (result[BASE_SCORE] - mean) / std
    result["base_score_rank_pct_within_timestamp"] = grouped.rank(method="average", pct=True)
    result["candidate_group_size"] = grouped.transform("size").astype(np.float32)
    for feature in META_FEATURES:
        result[feature] = pd.to_numeric(result[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return result


def _base_keep_columns() -> list[str]:
    return [*IDENTITY, "execution_decision_utc", "execution_label_end_utc", "candidate_month", TARGET, BASE_TARGET, BASE_SCORE, "base_oof_fold_start_utc", "base_oof_train_cutoff_utc"]


def generate_base_oof(labels: pd.DataFrame, raw_features: Sequence[str], *, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    first_eval = start + pd.Timedelta(days=BASE_WARMUP_DAYS)
    for side in SIDES:
        side_rows = labels.loc[labels["side_name"].eq(side)].copy()
        for fold_start in weekly_boundaries(first_eval, end):
            fold_end = min(fold_start + pd.Timedelta(days=FOLD_DAYS), end)
            train = side_rows.loc[side_rows["execution_label_end_utc"] <= fold_start].copy()
            evaluation = side_rows.loc[(side_rows["__ts__"] >= fold_start) & (side_rows["__ts__"] < fold_end)].copy()
            audit: dict[str, Any] = {"side": side, "fold_start_utc": fold_start, "fold_end_utc": fold_end, "train_rows": int(len(train)), "eval_rows": int(len(evaluation)), "max_train_label_end_utc": train["execution_label_end_utc"].max() if len(train) else None, "status": "trained"}
            if len(train) < MIN_BASE_TRAIN_ROWS or evaluation.empty:
                audit["status"] = "insufficient_prior_resolved_history"
                audits.append(audit)
                continue
            selected, importance = select_features(train, raw_features)
            fit = deterministic_fit_sample(train.sort_values(["__ts__", "__symbol__"], kind="stable"), MAX_FIT_ROWS)
            x_train, x_eval, medians = _fit_matrix(fit, evaluation, selected)
            model = _model(direct=False)
            model.fit(x_train, fit[BASE_TARGET].to_numpy(np.float32))
            evaluation[BASE_SCORE] = model.predict(x_eval).astype(np.float32)
            evaluation["base_oof_fold_start_utc"] = fold_start
            evaluation["base_oof_train_cutoff_utc"] = fold_start
            outputs.append(evaluation.loc[:, _base_keep_columns()])
            audit.update({"fit_rows": int(len(fit)), "selected_features": selected, "selection_abs_spearman": {name: importance[name] for name in selected}, "median_fill": medians})
            audits.append(audit)
    if not outputs:
        raise ValueError("no side-local base OOF folds trained")
    return add_candidate_context(pd.concat(outputs, ignore_index=True)), audits


def _validate_inner_base_oof(base_oof: pd.DataFrame) -> None:
    required = {BASE_SCORE, "base_oof_fold_start_utc", "base_oof_train_cutoff_utc", "execution_label_end_utc", *IDENTITY}
    missing = required - set(base_oof.columns)
    if missing:
        raise ValueError(f"meta layer missing required inner base OOF columns: {sorted(missing)}")
    if base_oof[BASE_SCORE].isna().any():
        raise ValueError("meta layer received missing base OOF scores")
    if (pd.to_datetime(base_oof["base_oof_train_cutoff_utc"], utc=True) > pd.to_datetime(base_oof["__ts__"], utc=True)).any():
        raise ValueError("base OOF cutoff is after its scored timestamp")


def generate_execution_ev_oof(base_oof: pd.DataFrame, *, evaluation_start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fit EV only on resolved, prior rows with strict inner base OOF scores."""

    _validate_inner_base_oof(base_oof)
    outputs: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for side in SIDES:
        side_rows = base_oof.loc[base_oof["side_name"].eq(side)].copy()
        for fold_start in weekly_boundaries(evaluation_start, end):
            fold_end = min(fold_start + pd.Timedelta(days=FOLD_DAYS), end)
            train = side_rows.loc[side_rows["execution_label_end_utc"] <= fold_start].copy()
            evaluation = side_rows.loc[(side_rows["__ts__"] >= fold_start) & (side_rows["__ts__"] < fold_end)].copy()
            audit: dict[str, Any] = {"side": side, "fold_start_utc": fold_start, "fold_end_utc": fold_end, "train_rows": int(len(train)), "eval_rows": int(len(evaluation)), "max_train_label_end_utc": train["execution_label_end_utc"].max() if len(train) else None, "status": "trained"}
            if len(train) < MIN_META_TRAIN_ROWS or evaluation.empty:
                audit["status"] = "insufficient_prior_resolved_base_oof_history"
                audits.append(audit)
                continue
            fit = deterministic_fit_sample(train.sort_values(["__ts__", "__symbol__"], kind="stable"), MAX_FIT_ROWS)
            x_train, x_eval, medians = _fit_matrix(fit, evaluation, META_FEATURES)
            model = _model(direct=True)
            model.fit(x_train, fit[TARGET].to_numpy(np.float32))
            evaluation[DIRECT_SCORE] = model.predict(x_eval).astype(np.float32)
            evaluation["execution_ev_oof_fold_start_utc"] = fold_start
            evaluation["execution_ev_oof_train_cutoff_utc"] = fold_start
            outputs.append(evaluation)
            audit.update({"fit_rows": int(len(fit)), "median_fill": medians, "meta_features": list(META_FEATURES)})
            audits.append(audit)
    if not outputs:
        raise ValueError("no strict two-layer execution-EV OOF folds trained")
    result = pd.concat(outputs, ignore_index=True).sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    if (result["__ts__"] < evaluation_start).any():
        raise ValueError("reported EV OOF contains warm-up rows")
    return result, audits


def top10_global_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Global top 10%, never per-timestamp or per-side admission."""

    def select(group: pd.DataFrame) -> pd.DataFrame:
        n = max(1, int(np.ceil(0.10 * len(group))))
        return group.nlargest(n, DIRECT_SCORE)

    def summarize(selected: pd.DataFrame, *, candidate_rows: int) -> dict[str, Any]:
        return {
            "candidate_rows": int(candidate_rows),
            "global_top10_rows": int(len(selected)),
            "global_top10_mean_net_ev_bps": float(selected[TARGET].mean() * 1e4),
            "global_top10_sum_net_return": float(selected[TARGET].sum()),
            "global_top10_positive_rate": float((selected[TARGET] > 0.0).mean()),
            "long_rows": int(selected["side_name"].eq("long").sum()),
            "short_rows": int(selected["side_name"].eq("short").sum()),
        }

    global_selected = select(frame)
    return {
        "global": summarize(global_selected, candidate_rows=len(frame)),
        "global_book_by_month": {
            str(month): summarize(
                group,
                candidate_rows=int(frame["candidate_month"].eq(month).sum()),
            )
            for month, group in global_selected.groupby("candidate_month", sort=True)
        },
        "global_book_by_side": {
            str(side): summarize(
                group, candidate_rows=int(frame["side_name"].eq(side).sum())
            )
            for side, group in global_selected.groupby("side_name", sort=True)
        },
        "diagnostic_month_local_pooled_top10": {
            str(month): summarize(select(group), candidate_rows=len(group))
            for month, group in frame.groupby("candidate_month", sort=True)
        },
    }


def comparator_manifest_contract() -> dict[str, Any]:
    return {
        "evidence_tier": "hourly_bar_approximation",
        "permitted_use": "historical regime/comparator diagnosis only",
        "forbidden_claims": [
            "exact 1m exit-geometry parity",
            "entry-timing or wait-action parity",
            "historical L2/spread/depth parity",
            "pooling metrics with exact-1m tier",
            "incumbent old55 score parity",
        ],
        "cost": "current side-parent round-trip fee exactly once; no historical spread inference",
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    warmup_start = pd.Timestamp(args.warmup_start, tz="UTC")
    evaluation_start = pd.Timestamp(args.evaluation_start, tz="UTC")
    evaluation_end = pd.Timestamp(args.evaluation_end, tz="UTC")
    if not (warmup_start < evaluation_start < evaluation_end):
        raise ValueError("require warmup_start < evaluation_start < evaluation_end")
    if evaluation_start != pd.Timestamp("2024-10-01T00:00:00Z") or evaluation_end != pd.Timestamp("2025-01-01T00:00:00Z"):
        raise ValueError("this comparator is intentionally fixed to Oct-Dec 2024 reporting")
    args.output_dir.mkdir(parents=True)
    universe, raw_features, schema_report = load_pit_candidate_universe(args.features_root, start=warmup_start, end=evaluation_end, minimum_features=args.minimum_raw_features)
    labels, coverage, geometry = materialize_hourly_labels(universe, hourly_root=args.hourly_root, policy=args.policy)
    base_oof, base_audit = generate_base_oof(labels, raw_features, start=warmup_start, end=evaluation_end)
    direct_oof, direct_audit = generate_execution_ev_oof(base_oof, evaluation_start=evaluation_start, end=evaluation_end)

    labels_path = args.output_dir / "hourly_execution_ev_12h_labels.parquet"
    oof_path = args.output_dir / "hourly_two_layer_execution_ev_strict_oof.parquet"
    coverage_path = args.output_dir / "hourly_coverage_by_month.csv"
    base_audit_path = args.output_dir / "base_fold_audit.json"
    direct_audit_path = args.output_dir / "execution_ev_fold_audit.json"
    label_keep = [*IDENTITY, "execution_decision_utc", "execution_label_end_utc", "candidate_month", "atr_fraction_14h_at_signal", "execution_gross_ev_12h", TARGET, "execution_cost_return", "execution_exit_reason", "execution_exit_hour", "execution_mfe_return_12h", "execution_mae_return_12h", BASE_TARGET]
    labels.loc[:, label_keep].to_parquet(labels_path, index=False, compression="zstd")
    oof_keep = [*IDENTITY, "execution_decision_utc", "execution_label_end_utc", "candidate_month", TARGET, BASE_TARGET, *META_FEATURES, DIRECT_SCORE, "base_oof_fold_start_utc", "base_oof_train_cutoff_utc", "execution_ev_oof_fold_start_utc", "execution_ev_oof_train_cutoff_utc"]
    direct_oof.loc[:, list(dict.fromkeys(oof_keep))].to_parquet(oof_path, index=False, compression="zstd")
    coverage_table = coverage.groupby("candidate_month", sort=True).agg(
        pit_universe_rows=("__symbol__", "size"),
        complete_hourly_paths=("complete_hourly_path", "sum"),
        complete_causal_atr=("complete_causal_atr", "sum"),
        complete_hourly_labels=("complete_hourly_label", "sum"),
    ).reset_index()
    coverage_table["missing_hourly_paths"] = (
        coverage_table["pit_universe_rows"] - coverage_table["complete_hourly_paths"]
    )
    coverage_table["hourly_path_coverage"] = coverage_table["complete_hourly_paths"] / coverage_table["pit_universe_rows"].clip(lower=1)
    coverage_table["causal_atr_coverage"] = coverage_table["complete_causal_atr"] / coverage_table["pit_universe_rows"].clip(lower=1)
    coverage_table["hourly_label_coverage"] = coverage_table["complete_hourly_labels"] / coverage_table["pit_universe_rows"].clip(lower=1)
    coverage_table.to_csv(coverage_path, index=False)
    _write_json(base_audit_path, {"folds": base_audit})
    _write_json(direct_audit_path, {"folds": direct_audit})
    artifacts = {path.name: _sha256(path) for path in (labels_path, oof_path, coverage_path, base_audit_path, direct_audit_path)}
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "period": {"warmup_training_start": warmup_start, "reported_oof_start": evaluation_start, "reported_oof_end_exclusive": evaluation_end, "actual_oof_start": direct_oof["__ts__"].min(), "actual_oof_end": direct_oof["__ts__"].max()},
        "rows": {"pit_universe": int(len(universe)), "hourly_labels": int(len(labels)), "base_oof": int(len(base_oof)), "reported_two_layer_oof": int(len(direct_oof))},
        "source": {
            "feature_store": str(args.features_root),
            "feature_schema": schema_report,
            "candidate_universe": "all rows physically available in the cached historical feature store at each timestamp; one candidate per side; no later score, liquidity or candidate screen",
            "pit_parity_status": "diagnostic cache: sample historical transforms have not been independently recomputed bitwise from truncated raw history",
            "hourly_ohlcv": str(args.hourly_root),
        },
        "raw_features": raw_features,
        "target": {"name": TARGET, "simulator": "simulate_execution_ev_12h", "bar_minutes": 60, "horizon_hours": HORIZON_HOURS, "signal_to_decision_hours": DECISION_DELAY_HOURS, "first_path_bar": "next hourly executable bar", "atr": "14 completed hourly bars ending at the signal timestamp", "cost": "current side-parent round-trip fee exactly once", "geometry": geometry, "policy": str(args.policy), "policy_sha256": _sha256(args.policy)},
        "base": {"target": BASE_TARGET, "definition": f"sigmoid({TARGET}/{SOFT_LABEL_SCALE_RETURN})", "side_local": True, "walk_forward": f"{FOLD_DAYS}d expanding", "warmup_days": BASE_WARMUP_DAYS, "feature_selection": f"fold-local top {MAX_SELECTED_FEATURES} absolute Spearman from configured raw/PIT pool", "imputation": "training-fold median only"},
        "execution_ev": {"target": TARGET, "side_local": True, "inputs": list(META_FEATURES), "base_score_provenance": "strict inner expanding OOF only", "purge_hours": HORIZON_HOURS, "imputation": "training-fold median only"},
        "validation": {
            "ranking": "one pooled global top 10% after side-local common-unit execution EV; month/side metrics slice that frozen book",
            "warmup": "Jul-Sep 2024 base/inner-meta history only; no Oct-Dec outcome trains an earlier fold",
            "fold_local": "feature selection and imputation are refit only on prior resolved rows",
            "representation_selection_exception": "the configured current raw feature family is later-selected; outcomes are excluded and supervised selection remains fold-local, but this is diagnostic rather than untouched promotion evidence",
        },
        "comparator_limits": comparator_manifest_contract(),
        "coverage": coverage_table.to_dict(orient="records"),
        "metrics": top10_global_metrics(direct_oof),
        "artifacts": artifacts,
    }
    summary["manifest_sha256"] = _canonical_hash(summary)
    summary_path = args.output_dir / "summary.json"
    _write_json(summary_path, summary)
    return {"labels": labels_path, "oof": oof_path, "coverage": coverage_path, "summary": summary_path}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--features-root", type=Path, default=ROOT / "data_perp/features/20260711_070000")
    result.add_argument("--hourly-root", type=Path, default=ROOT / "data_perp/exchanges/krakenfutures")
    result.add_argument("--policy", type=Path, default=ROOT / "data_perp/reports/simple_policy_1m_joint_trailing_raw_bayesian_champion_20260718_v1/production_staging/best_policy_params.json")
    result.add_argument("--warmup-start", default="2024-07-01")
    result.add_argument("--evaluation-start", default="2024-10-01")
    result.add_argument("--evaluation-end", default="2025-01-01")
    result.add_argument("--minimum-raw-features", type=int, default=12)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.minimum_raw_features < 1:
        raise ValueError("--minimum-raw-features must be positive")
    print(json.dumps({key: str(value) for key, value in run(arguments).items()}, indent=2))
