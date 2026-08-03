#!/usr/bin/env python3
"""Materialize exact 1m h12 clean-first labels for the frozen July cohort.

This is a target-only materializer.  It binds the frozen July candidate
identities, decision-time geometry and immutable Kraken 1m store, then applies
the same h12_u1p5atr competing-risk definition used by the cross-era data:
the favourable barrier is ``max(1.5 * ATR_fraction, 1.5%)`` and the adverse
barrier is ``1.0 * ATR_fraction``.  The side-specific executable entry and
exit half-spreads are applied before evaluating each barrier, exactly as in
the historical exact-1m path-head materializer.  A same-minute OHLC conflict
is assigned to the adverse event deterministically.  It does not modify policy
labels, features, scores, or models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import canonical_kraken_execution_1m_root  # noqa: E402
from scripts.materialize_execution_ev_policy_labels import (  # noqa: E402
    PATH_COLUMNS,
    _load_symbol_bars,
)


SCHEMA = "july_exact1m_clean_first_labels_v1"
IDENTITY = ("__ts__", "__symbol__", "side_name", "candidate_id")
HORIZON_MINUTES = 720
UPPER_ATR = 1.5
UPPER_RETURN_FLOOR = 0.015
LOWER_ATR = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _first_index(mask: np.ndarray) -> np.ndarray:
    """Return the zero-based first true index, or -1 where no event occurs."""

    if mask.ndim != 2:
        raise ValueError("event masks must be two-dimensional")
    return np.where(mask.any(axis=1), mask.argmax(axis=1), -1).astype(np.int16)


def build_exact_clean_first_labels(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    *,
    atr_fraction: np.ndarray,
    side_sign: np.ndarray,
    entry_half_spread_bps: np.ndarray,
    exit_half_spread_bps: np.ndarray,
    decision_utc: Sequence[Any],
) -> pd.DataFrame:
    """Apply the h12_u1p5atr event contract to complete exact 1m OHLC paths."""

    arrays = tuple(np.asarray(values, dtype=np.float64) for values in (open_, high, low))
    if any(values.ndim != 2 for values in arrays):
        raise ValueError("exact 1m OHLC paths must be two-dimensional")
    if not (open_.shape == high.shape == low.shape):
        raise ValueError("open/high/low paths must have the same shape")
    rows, minutes = open_.shape
    if minutes != HORIZON_MINUTES:
        raise ValueError(f"exact clean-first paths must contain {HORIZON_MINUTES} minutes")
    atr = np.asarray(atr_fraction, dtype=np.float64)
    sign = np.asarray(side_sign, dtype=np.float64)
    entry_spread = np.asarray(entry_half_spread_bps, dtype=np.float64)
    exit_spread = np.asarray(exit_half_spread_bps, dtype=np.float64)
    decision = pd.to_datetime(decision_utc, utc=True, errors="raise")
    if (
        atr.shape != (rows,)
        or sign.shape != (rows,)
        or entry_spread.shape != (rows,)
        or exit_spread.shape != (rows,)
        or decision.shape != (rows,)
    ):
        raise ValueError("event vectors must have one value per exact path")
    if (
        not np.isfinite(np.stack(arrays, axis=2)).all()
        or not np.isfinite(atr).all()
        or not np.isfinite(entry_spread).all()
        or not np.isfinite(exit_spread).all()
        or (arrays[0] <= 0.0).any()
        or (arrays[1] <= 0.0).any()
        or (arrays[2] <= 0.0).any()
        or (atr <= 0.0).any()
        or (entry_spread < 0.0).any()
        or (exit_spread < 0.0).any()
    ):
        raise ValueError("exact paths and ATR fractions must be finite and strictly positive")
    if not np.isin(sign, (-1.0, 1.0)).all():
        raise ValueError("side sign must be canonical -1/+1")

    raw_entry = arrays[0][:, 0]
    # Exact parity with materialize_febapr2025_exact1m_path_head_labels.py:
    # entry is executable at the entry half-spread; all future exit candidates
    # are executable at the corresponding side-specific exit half-spread.
    entry = raw_entry * (1.0 + sign * entry_spread / 10_000.0)
    executable_high = arrays[1] * (1.0 - sign[:, None] * exit_spread[:, None] / 10_000.0)
    executable_low = arrays[2] * (1.0 - sign[:, None] * exit_spread[:, None] / 10_000.0)
    upper = np.maximum(UPPER_ATR * atr, UPPER_RETURN_FLOOR)
    lower = LOWER_ATR * atr
    long_favorable = executable_high / entry[:, None] - 1.0 >= upper[:, None]
    long_adverse = 1.0 - executable_low / entry[:, None] >= lower[:, None]
    short_favorable = 1.0 - executable_low / entry[:, None] >= upper[:, None]
    short_adverse = executable_high / entry[:, None] - 1.0 >= lower[:, None]
    favorable_mask = np.where(sign[:, None] > 0.0, long_favorable, short_favorable)
    adverse_mask = np.where(sign[:, None] > 0.0, long_adverse, short_adverse)
    first_favorable = _first_index(favorable_mask)
    first_adverse = _first_index(adverse_mask)
    has_favorable, has_adverse = first_favorable >= 0, first_adverse >= 0
    favorable_first = has_favorable & (~has_adverse | (first_favorable < first_adverse))
    adverse_first = has_adverse & (~has_favorable | (first_adverse <= first_favorable))
    timeout = ~favorable_first & ~adverse_first
    if not np.all(favorable_first.astype(int) + adverse_first.astype(int) + timeout.astype(int) == 1):
        raise AssertionError("competing-risk events are not exhaustive and exclusive")
    conflict = has_favorable & has_adverse & (first_favorable == first_adverse)
    event = np.full(rows, "timeout", dtype=object)
    event[favorable_first] = "favorable_first"
    event[adverse_first] = "adverse_first_or_conflict"
    first_favorable_utc = decision + pd.to_timedelta(np.where(has_favorable, first_favorable, 0), unit="min")
    first_adverse_utc = decision + pd.to_timedelta(np.where(has_adverse, first_adverse, 0), unit="min")
    first_favorable_utc = first_favorable_utc.where(has_favorable, pd.NaT)
    first_adverse_utc = first_adverse_utc.where(has_adverse, pd.NaT)
    label_end = decision + pd.Timedelta(minutes=HORIZON_MINUTES)
    return pd.DataFrame(
        {
            "__soft_tb_upper_return__": upper.astype(np.float32),
            "__soft_tb_lower_return__": lower.astype(np.float32),
            "__soft_tb_raw_entry_open__": raw_entry.astype(np.float64),
            "__soft_tb_executable_entry__": entry.astype(np.float64),
            "__soft_tb_upper_hit_12h__": has_favorable.astype(np.int8),
            "__soft_tb_lower_hit_12h__": has_adverse.astype(np.int8),
            "__soft_tb_first_event__": pd.array(event, dtype="string"),
            "__soft_tb_order_ambiguous__": conflict.astype(np.int8),
            "__soft_tb_first_favorable_minute__": np.where(has_favorable, first_favorable, np.nan).astype(np.float32),
            "__soft_tb_first_adverse_minute__": np.where(has_adverse, first_adverse, np.nan).astype(np.float32),
            "__soft_tb_first_favorable_utc__": first_favorable_utc.to_numpy(),
            "__soft_tb_first_adverse_utc__": first_adverse_utc.to_numpy(),
            "__soft_tb_label_end_utc__": label_end.to_numpy(),
            "__soft_tb_label_available_at__": label_end.to_numpy(),
        }
    )


def _load_july_source(candidates_path: Path, path_targets_path: Path, policy_labels_path: Path) -> pd.DataFrame:
    candidates = pd.read_parquet(candidates_path, columns=list(IDENTITY))
    targets = pd.read_parquet(
        path_targets_path,
        columns=[*IDENTITY, "__path_auxiliary_atr_fraction__", "__barrier_pct__"],
    )
    policy = pd.read_parquet(
        policy_labels_path,
        columns=[
            *IDENTITY,
            "execution_decision_utc",
            "execution_label_end_utc",
            "execution_entry_half_spread_bps",
            "execution_exit_half_spread_bps",
            "execution_cost_return",
        ],
    )
    for name, frame in (("candidates", candidates), ("path targets", targets), ("policy labels", policy)):
        if frame.duplicated(list(IDENTITY), keep=False).any():
            raise ValueError(f"{name} contain duplicate exact identities")
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        frame["side_name"] = frame["side_name"].astype(str).str.lower()
        if not frame["side_name"].isin(("long", "short")).all():
            raise ValueError(f"{name} have non-canonical side names")
    # The feature surface is intentionally wider (14,400 rows) than the frozen
    # decision cohort.  Geometry defines the exact 5,760-row label universe;
    # assert that every one of those identities is present in the feature source
    # without accidentally expanding the target cohort.
    candidate_keys = candidates.loc[:, list(IDENTITY)]
    source = targets.merge(
        candidate_keys,
        on=list(IDENTITY),
        how="left",
        validate="one_to_one",
        indicator="__candidate_identity_present__",
    )
    if not source["__candidate_identity_present__"].eq("both").all():
        raise ValueError("frozen geometry contains identities absent from the candidate feature surface")
    source = source.drop(columns="__candidate_identity_present__").merge(
        policy, on=list(IDENTITY), how="left", validate="one_to_one"
    )
    required = (
        "__path_auxiliary_atr_fraction__",
        "__barrier_pct__",
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_entry_half_spread_bps",
        "execution_exit_half_spread_bps",
        "execution_cost_return",
    )
    if source.loc[:, list(required)].isna().any().any():
        missing = {column: int(source[column].isna().sum()) for column in required if source[column].isna().any()}
        raise ValueError(f"frozen July identity join is incomplete: {missing}")
    source["execution_decision_utc"] = pd.to_datetime(source["execution_decision_utc"], utc=True, errors="raise")
    source["execution_label_end_utc"] = pd.to_datetime(source["execution_label_end_utc"], utc=True, errors="raise")
    expected_end = source["execution_decision_utc"] + pd.Timedelta(minutes=HORIZON_MINUTES)
    if not source["execution_label_end_utc"].eq(expected_end).all():
        raise ValueError("policy labels do not bind the required exact 12h decision horizon")
    for column in (
        "__path_auxiliary_atr_fraction__",
        "__barrier_pct__",
        "execution_entry_half_spread_bps",
        "execution_exit_half_spread_bps",
        "execution_cost_return",
    ):
        values = pd.to_numeric(source[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must be finite")
        if column in {"__path_auxiliary_atr_fraction__", "__barrier_pct__"} and (values <= 0.0).any():
            raise ValueError(f"{column} must be strictly positive")
        if column in {"execution_entry_half_spread_bps", "execution_exit_half_spread_bps", "execution_cost_return"} and (values < 0.0).any():
            raise ValueError(f"{column} must be non-negative")
    return source.sort_values(["execution_decision_utc", "__symbol__", "side_name"], kind="stable").reset_index(drop=True)


def _load_historical_feature_source(feature_universe_path: Path, policy_labels_path: Path) -> pd.DataFrame:
    """Bind the May--July feature universe to the exact 1m spread contract.

    The historical feature universe is the declared 134,889-row training
    cohort.  It already carries decision time and raw ATR, while the exact
    deployed-policy ledger supplies immutable executable half-spreads.  Both
    decision and label-end timestamps must agree exactly before any path is
    read.
    """

    features = pd.read_parquet(
        feature_universe_path,
        columns=[
            *IDENTITY,
            "oof_entry_atr_fraction",
            "execution_decision_utc",
            "execution_label_end_utc",
            "execution_cost_return",
        ],
    )
    policy = pd.read_parquet(
        policy_labels_path,
        columns=[
            *IDENTITY,
            "execution_decision_utc",
            "execution_label_end_utc",
            "execution_entry_half_spread_bps",
            "execution_exit_half_spread_bps",
            "execution_cost_return",
        ],
    )
    for name, frame in (("feature universe", features), ("policy labels", policy)):
        if frame.duplicated(list(IDENTITY), keep=False).any():
            raise ValueError(f"{name} contain duplicate exact identities")
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        frame["side_name"] = frame["side_name"].astype(str).str.lower()
        if not frame["side_name"].isin(("long", "short")).all():
            raise ValueError(f"{name} have non-canonical side names")
    source = features.merge(policy, on=list(IDENTITY), how="left", validate="one_to_one", suffixes=("_feature", "_policy"))
    required_policy = (
        "execution_decision_utc_policy",
        "execution_label_end_utc_policy",
        "execution_entry_half_spread_bps",
        "execution_exit_half_spread_bps",
        "execution_cost_return_policy",
    )
    if source.loc[:, list(required_policy)].isna().any().any():
        missing = {column: int(source[column].isna().sum()) for column in required_policy if source[column].isna().any()}
        raise ValueError(f"historical feature universe does not have complete exact-policy lineage: {missing}")
    for stem in ("execution_decision_utc", "execution_label_end_utc"):
        left = pd.to_datetime(source[f"{stem}_feature"], utc=True, errors="raise")
        right = pd.to_datetime(source[f"{stem}_policy"], utc=True, errors="raise")
        if not left.eq(right).all():
            raise ValueError(f"historical feature and exact-policy {stem} disagree")
        source[stem] = right
    feature_cost = pd.to_numeric(source["execution_cost_return_feature"], errors="coerce").to_numpy(float)
    policy_cost = pd.to_numeric(source["execution_cost_return_policy"], errors="coerce").to_numpy(float)
    if not np.isfinite(feature_cost).all() or not np.isfinite(policy_cost).all() or not np.allclose(feature_cost, policy_cost, rtol=0.0, atol=1e-12):
        raise ValueError("historical feature and exact-policy fee returns disagree")
    source["execution_cost_return"] = policy_cost
    source["__path_auxiliary_atr_fraction__"] = pd.to_numeric(source["oof_entry_atr_fraction"], errors="coerce")
    source["__barrier_pct__"] = np.nan
    for column in (
        "__path_auxiliary_atr_fraction__",
        "execution_entry_half_spread_bps",
        "execution_exit_half_spread_bps",
        "execution_cost_return",
    ):
        values = pd.to_numeric(source[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(f"historical {column} must be finite and non-negative")
    if (source["__path_auxiliary_atr_fraction__"].to_numpy(float) <= 0.0).any():
        raise ValueError("historical ATR fraction must be strictly positive")
    expected_end = source["execution_decision_utc"] + pd.Timedelta(minutes=HORIZON_MINUTES)
    if not source["execution_label_end_utc"].eq(expected_end).all():
        raise ValueError("historical exact-policy labels do not bind the required 12h horizon")
    return source.loc[:, [
        *IDENTITY,
        "__path_auxiliary_atr_fraction__",
        "__barrier_pct__",
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_entry_half_spread_bps",
        "execution_exit_half_spread_bps",
        "execution_cost_return",
    ]].sort_values(["execution_decision_utc", "__symbol__", "side_name"], kind="stable").reset_index(drop=True)


def _materialize_symbol(source: pd.DataFrame, data_root: Path, *, batch_rows: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    symbol = str(source["__symbol__"].iloc[0])
    start = pd.Timestamp(source["execution_decision_utc"].min())
    end = pd.Timestamp(source["execution_decision_utc"].max()) + pd.Timedelta(minutes=HORIZON_MINUTES)
    bars = _load_symbol_bars(data_root, symbol, start, end)
    grid = pd.date_range(start, end, freq="min", inclusive="left", tz="UTC")
    dense = bars.reindex(grid).loc[:, list(PATH_COLUMNS)]
    values = dense.to_numpy(dtype=np.float64)
    offsets = ((source["execution_decision_utc"] - start) / pd.Timedelta(minutes=1)).astype(np.int64).to_numpy()
    good = np.isfinite(values).all(axis=1) & (values > 0.0).all(axis=1) & (values[:, 1] >= values[:, 2])
    prefix = np.concatenate(([0], np.cumsum(good, dtype=np.int64)))
    complete = prefix[offsets + HORIZON_MINUTES] - prefix[offsets] == HORIZON_MINUTES
    if not complete.all():
        raise ValueError(f"immutable exact 1m path coverage is incomplete for {symbol}: {int((~complete).sum())}")
    parts: list[pd.DataFrame] = []
    for begin in range(0, len(source), int(batch_rows)):
        stop = min(begin + int(batch_rows), len(source))
        batch = source.iloc[begin:stop].reset_index(drop=True)
        local_offsets = offsets[begin:stop]
        matrices = tuple(
            np.stack([values[offset : offset + HORIZON_MINUTES, index] for offset in local_offsets])
            for index in range(len(PATH_COLUMNS))
        )
        labels = build_exact_clean_first_labels(
            *matrices[:3],
            atr_fraction=batch["__path_auxiliary_atr_fraction__"].to_numpy(float),
            side_sign=np.where(batch["side_name"].eq("long"), 1.0, -1.0),
            entry_half_spread_bps=batch["execution_entry_half_spread_bps"].to_numpy(float),
            exit_half_spread_bps=batch["execution_exit_half_spread_bps"].to_numpy(float),
            decision_utc=batch["execution_decision_utc"],
        )
        parts.append(pd.concat([batch, labels], axis=1))
    return pd.concat(parts, ignore_index=True), {
        "__symbol__": symbol,
        "rows": int(len(source)),
        "complete_rows": int(complete.sum()),
        "coverage": float(complete.mean()),
    }


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["month"] = pd.to_datetime(work["__ts__"], utc=True).dt.strftime("%Y-%m")
    return (
        work.groupby(["month", "side_name"], observed=True, sort=True)
        .agg(
            rows=("candidate_id", "size"),
            favorable_first_rate=("__soft_tb_first_event__", lambda values: float(values.eq("favorable_first").mean())),
            adverse_first_or_conflict_rate=("__soft_tb_first_event__", lambda values: float(values.eq("adverse_first_or_conflict").mean())),
            timeout_rate=("__soft_tb_first_event__", lambda values: float(values.eq("timeout").mean())),
            same_minute_conflict_rows=("__soft_tb_order_ambiguous__", "sum"),
            mean_upper_return=("__soft_tb_upper_return__", "mean"),
            mean_lower_return=("__soft_tb_lower_return__", "mean"),
        )
        .reset_index()
    )


def materialize(
    *,
    candidates_path: Path,
    path_targets_path: Path,
    policy_labels_path: Path,
    policy_labels_manifest_path: Path,
    data_root: Path,
    output_dir: Path,
    batch_rows: int = 512,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    source = _load_july_source(candidates_path, path_targets_path, policy_labels_path)
    stage = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    try:
        parts: list[pd.DataFrame] = []
        coverage_rows: list[dict[str, Any]] = []
        for _, positions in source.groupby("__symbol__", sort=True).indices.items():
            labels, coverage = _materialize_symbol(source.iloc[np.asarray(positions, dtype=np.int64)].reset_index(drop=True), data_root, batch_rows=batch_rows)
            parts.append(labels)
            coverage_rows.append(coverage)
        result = pd.concat(parts, ignore_index=True).sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
        if len(result) != len(source) or result.duplicated(list(IDENTITY), keep=False).any():
            raise AssertionError("exact label output did not preserve the frozen candidate identity set")
        event_columns = ["__soft_tb_first_event__", "__soft_tb_first_favorable_utc__", "__soft_tb_first_adverse_utc__"]
        if result.loc[:, event_columns].isna().all(axis=1).any():
            raise AssertionError("all event/timestamp fields are missing for a candidate")
        labels_path = stage / "exact_clean_first_labels.parquet"
        coverage_path = stage / "coverage_by_symbol.csv"
        summary_path = stage / "support_by_month_side.csv"
        result.to_parquet(labels_path, index=False, compression="zstd")
        coverage = pd.DataFrame(coverage_rows).sort_values("__symbol__", kind="stable")
        coverage.to_csv(coverage_path, index=False)
        support = _summary(result)
        support.to_csv(summary_path, index=False)
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            revision = "unknown"
        manifest = {
            "schema": SCHEMA,
            "status": "completed_exact_1m_target_only_not_model_evidence",
            "rows": int(len(result)),
            "identity": list(IDENTITY),
            "identity_unique": True,
            "lineage": {
                "candidates": {"path": str(candidates_path), "sha256": _sha256(candidates_path)},
                "frozen_decision_geometry": {"path": str(path_targets_path), "sha256": _sha256(path_targets_path)},
                "exact_policy_labels": {"path": str(policy_labels_path), "sha256": _sha256(policy_labels_path)},
                "exact_policy_labels_manifest": {"path": str(policy_labels_manifest_path), "sha256": _sha256(policy_labels_manifest_path)},
                "immutable_execution_1m_root": str(canonical_kraken_execution_1m_root(data_root)),
            },
            "path_contract": {
                "cadence_minutes": 1,
                "path_minutes": HORIZON_MINUTES,
                "path_start": "raw immutable 1m decision-open bar at execution_decision_utc",
                "path_end": "exclusive decision + 12h",
                "complete_coverage_required": True,
                "interpolation_or_asof_fill": "forbidden",
            },
            "event_contract": {
                "grid_name": "h12_u1p5atr",
                "upper": "max(1.5 * decision-time ATR fraction, 1.5% cost-aware meaningful floor)",
                "lower": "1.0 * decision-time ATR fraction",
                "orientation": "side-relative, long/short mirrored",
                "executable_price_adjustment": "entry open * (1 + side * entry_half_spread_bps/10000); future high/low * (1 - side * exit_half_spread_bps/10000), exactly as Feb-Apr exact-1m path-head labels",
                "fee": "execution_cost_return is source-bound for accounting parity; it is not a second barrier deduction because the h12_u1p5atr event contract uses the cost-aware 1.5% meaningful floor",
                "same_minute_ohlc_conflict": "adverse_first_or_conflict",
                "first_event": "favorable_first, adverse_first_or_conflict, or timeout; mutually exclusive",
                "first_event_minute": "zero-based minute offset from exact execution decision timestamp",
                "label_available_at": "decision + 12h",
            },
            "coverage": {
                "complete_rows": int(coverage["complete_rows"].sum()),
                "rows": int(coverage["rows"].sum()),
                "rate": float(coverage["complete_rows"].sum() / max(coverage["rows"].sum(), 1)),
            },
            "outputs": {
                "labels": {"path": str(output_dir / labels_path.name), "sha256": _sha256(labels_path), "rows": int(len(result))},
                "coverage_by_symbol": {"path": str(output_dir / coverage_path.name), "sha256": _sha256(coverage_path)},
                "support_by_month_side": {"path": str(output_dir / summary_path.name), "sha256": _sha256(summary_path)},
            },
            "code_revision": revision,
        }
        _write_json(stage / "manifest.json", manifest)
        os.replace(stage, output_dir)
    except Exception:
        if stage.exists():
            import shutil
            shutil.rmtree(stage)
        raise
    return manifest


def _shard_name(position: int, symbol: str) -> str:
    return f"{position:04d}_{hashlib.sha256(symbol.encode()).hexdigest()[:16]}.parquet"


def _validate_resumable_shard(path: Path, expected: pd.DataFrame) -> None:
    actual = pd.read_parquet(path, columns=list(IDENTITY))
    for frame in (actual, expected):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    actual = actual.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    wanted = expected.loc[:, list(IDENTITY)].sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    if len(actual) != len(wanted) or not actual.equals(wanted):
        raise ValueError(f"resumable exact-label shard does not match its frozen identity slice: {path}")


def materialize_historical_feature_universe(
    *,
    feature_universe_path: Path,
    policy_labels_path: Path,
    policy_labels_manifest_path: Path,
    data_root: Path,
    output_dir: Path,
    batch_rows: int = 512,
    resume: bool = False,
) -> dict[str, Any]:
    """Resumably materialize the 134,889-row May--July exact1m training target.

    Shards are atomic per symbol.  A stopped job can only resume if each
    discovered shard exactly matches its frozen identity slice; final combined
    output and manifest are written only once all source identities are present.
    """

    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    source = _load_historical_feature_source(feature_universe_path, policy_labels_path)
    if output_dir.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if output_dir.exists() and (output_dir / "manifest.json").exists():
        raise FileExistsError(f"historical exact label artifact is already complete: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(exist_ok=True)
    coverage_rows: list[dict[str, Any]] = []
    grouped = list(source.groupby("__symbol__", sort=True).indices.items())
    for position, (symbol, indices) in enumerate(grouped):
        local = source.iloc[np.asarray(indices, dtype=np.int64)].reset_index(drop=True)
        shard_path = shards_dir / _shard_name(position, str(symbol))
        if shard_path.exists():
            if not resume:
                raise FileExistsError(f"existing shard requires --resume: {shard_path}")
            _validate_resumable_shard(shard_path, local)
            coverage_rows.append({"__symbol__": str(symbol), "rows": int(len(local)), "complete_rows": int(len(local)), "coverage": 1.0, "resumed": True})
            continue
        labels, coverage = _materialize_symbol(local, data_root, batch_rows=batch_rows)
        temporary = shard_path.with_name(f".{shard_path.name}.{uuid.uuid4().hex}.tmp")
        labels.to_parquet(temporary, index=False, compression="zstd")
        _validate_resumable_shard(temporary, local)
        os.replace(temporary, shard_path)
        coverage["resumed"] = False
        coverage_rows.append(coverage)
        _write_json(output_dir / "progress.json", {
            "schema": SCHEMA,
            "status": "resumable_in_progress",
            "source_rows": int(len(source)),
            "completed_symbols": int(len(coverage_rows)),
            "total_symbols": int(len(grouped)),
            "latest_symbol": str(symbol),
        })
    shard_paths = [shards_dir / _shard_name(position, str(symbol)) for position, (symbol, _) in enumerate(grouped)]
    result = pd.concat([pd.read_parquet(path) for path in shard_paths], ignore_index=True).sort_values(list(IDENTITY), kind="stable").reset_index(drop=True)
    if len(result) != len(source) or result.duplicated(list(IDENTITY), keep=False).any():
        raise AssertionError("resumable historical materialization did not preserve every frozen identity")
    labels_path = output_dir / "exact_clean_first_labels.parquet"
    coverage_path = output_dir / "coverage_by_symbol.csv"
    support_path = output_dir / "support_by_month_side.csv"
    result.to_parquet(labels_path, index=False, compression="zstd")
    coverage = pd.DataFrame(coverage_rows).sort_values("__symbol__", kind="stable")
    coverage.to_csv(coverage_path, index=False)
    support = _summary(result)
    support.to_csv(support_path, index=False)
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = "unknown"
    manifest = {
        "schema": SCHEMA,
        "status": "completed_resumable_exact_1m_harmonized_training_targets_not_model_evidence",
        "rows": int(len(result)),
        "identity": list(IDENTITY),
        "identity_unique": True,
        "lineage": {
            "feature_universe": {"path": str(feature_universe_path), "sha256": _sha256(feature_universe_path), "rows": int(len(source))},
            "exact_policy_labels": {"path": str(policy_labels_path), "sha256": _sha256(policy_labels_path)},
            "exact_policy_labels_manifest": {"path": str(policy_labels_manifest_path), "sha256": _sha256(policy_labels_manifest_path)},
            "immutable_execution_1m_root": str(canonical_kraken_execution_1m_root(data_root)),
        },
        "path_contract": {
            "cadence_minutes": 1,
            "path_minutes": HORIZON_MINUTES,
            "path_start": "raw immutable 1m decision-open bar at execution_decision_utc",
            "path_end": "exclusive decision + 12h",
            "complete_coverage_required": True,
            "interpolation_or_asof_fill": "forbidden",
            "resumable_sharding": "atomic symbol shards; existing shard identity must exactly match frozen source before reuse",
        },
        "event_contract": {
            "grid_name": "h12_u1p5atr",
            "upper": "max(1.5 * decision-time ATR fraction, 1.5% cost-aware meaningful floor)",
            "lower": "1.0 * decision-time ATR fraction",
            "orientation": "side-relative, long/short mirrored",
            "executable_price_adjustment": "entry open * (1 + side * entry_half_spread_bps/10000); future high/low * (1 - side * exit_half_spread_bps/10000), exactly as Feb-Apr exact-1m path-head labels",
            "same_minute_ohlc_conflict": "adverse_first_or_conflict",
            "fee": "exact policy fee is bound and reconciled before materialization; it is not double-deducted from the h12_u1p5atr barrier event",
            "label_available_at": "decision + 12h",
        },
        "coverage": {"complete_rows": int(coverage["complete_rows"].sum()), "rows": int(coverage["rows"].sum()), "rate": float(coverage["complete_rows"].sum() / max(coverage["rows"].sum(), 1))},
        "outputs": {
            "labels": {"path": str(labels_path), "sha256": _sha256(labels_path), "rows": int(len(result))},
            "coverage_by_symbol": {"path": str(coverage_path), "sha256": _sha256(coverage_path)},
            "support_by_month_side": {"path": str(support_path), "sha256": _sha256(support_path)},
            "shards": {"directory": str(shards_dir), "count": len(shard_paths)},
        },
        "code_revision": revision,
    }
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "progress.json").unlink(missing_ok=True)
    return manifest


def _parser() -> argparse.ArgumentParser:
    root = ROOT / "data_perp/artifacts/execution_ev_july20_23_retrospective_20260730_v2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("july_frozen", "historical_feature_universe"), default="july_frozen")
    parser.add_argument("--candidates", type=Path, default=root / "candidates/candidate_features.parquet")
    parser.add_argument("--path-targets", type=Path, default=root / "geometry/path_targets.parquet")
    parser.add_argument("--policy-labels", type=Path, default=root / "labels_12h/execution_ev_policy_labels.parquet")
    parser.add_argument("--policy-labels-manifest", type=Path, default=root / "labels_12h/manifest.json")
    parser.add_argument(
        "--historical-feature-universe",
        type=Path,
        default=ROOT / "data_perp/artifacts/exact_policy_capture_feature_universe_20260727_v2/capture_feature_universe.parquet",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data_perp")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data_perp/artifacts/july_exact1m_clean_first_labels_20260730_v1")
    parser.add_argument("--batch-rows", type=int, default=512)
    parser.add_argument("--resume", action="store_true", help="resume only validated atomic historical symbol shards")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "july_frozen":
        output = materialize(
            candidates_path=args.candidates,
            path_targets_path=args.path_targets,
            policy_labels_path=args.policy_labels,
            policy_labels_manifest_path=args.policy_labels_manifest,
            data_root=args.data_root,
            output_dir=args.output_dir,
            batch_rows=args.batch_rows,
        )
    else:
        output = materialize_historical_feature_universe(
            feature_universe_path=args.historical_feature_universe,
            policy_labels_path=args.policy_labels,
            policy_labels_manifest_path=args.policy_labels_manifest,
            data_root=args.data_root,
            output_dir=args.output_dir,
            batch_rows=args.batch_rows,
            resume=args.resume,
        )
    print(json.dumps(_json_safe(output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
