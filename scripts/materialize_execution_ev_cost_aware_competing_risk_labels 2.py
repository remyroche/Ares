#!/usr/bin/env python3
"""Materialize full-horizon, row-cost-aware exact 1m competing-risk labels.

This is deliberately a *target-only* diagnostic materializer.  It starts each
candidate at its exact executable decision entry and reads every one-minute bar
through ``decision + 12h``.  It does not use, or stop at, the deployed policy
exit.  The favourable barrier is candidate-local because it must clear both the
meaningful-MFE floor and that candidate's already-accounted execution cost.

The output has one row per frozen candidate identity and requested buffer.  It
is not a decision-time feature surface and it is not promotion evidence by
itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import (  # noqa: E402
    canonical_kraken_execution_1m_root,
)
from scripts.materialize_execution_ev_policy_labels import (  # noqa: E402
    IDENTITY,
    PATH_COLUMNS,
    _load_symbol_bars,
)


SCHEMA = "execution_ev_cost_aware_competing_risk_1m_labels_v1"
SOURCE_SCHEMA = "execution_ev_deployed_policy_1m_labels_v1"
HORIZON_MINUTES = 720
UPPER_ATR = 1.5
UPPER_RETURN_FLOOR = 0.015
LOWER_ATR = 1.0
DEFAULT_BUFFERS_BPS = (0, 25, 50, 100)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
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
    if mask.ndim != 2:
        raise ValueError("event masks must be two-dimensional")
    return np.where(mask.any(axis=1), mask.argmax(axis=1), -1).astype(np.int16)


def _finite_vector(name: str, values: Sequence[float], rows: int, *, positive: bool = False) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (rows,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector matching exact paths")
    if positive and (result <= 0.0).any():
        raise ValueError(f"{name} must be strictly positive")
    return result


def build_row_cost_aware_competing_risk_labels(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    oof_entry_atr_fraction: Sequence[float],
    execution_cost_return: Sequence[float],
    execution_entry_price: Sequence[float],
    side_sign: Sequence[float],
    decision_utc: Sequence[Any],
    buffer_bps: int,
    use_upper_return_floor: bool = True,
) -> pd.DataFrame:
    """Return mutually-exclusive full-horizon economic competing risks.

    ``execution_entry_price`` is the frozen executable policy entry, so the
    future OHLC path is evaluated in the same return units as row-level gross,
    cost and net execution labels.  A candidate's actual policy exit is never
    supplied to this function and cannot truncate the 720-minute event path.
    """

    arrays = tuple(np.asarray(values, dtype=np.float64) for values in (open_, high, low, close))
    if any(values.ndim != 2 for values in arrays):
        raise ValueError("exact 1m OHLC paths must be two-dimensional")
    if not (open_.shape == high.shape == low.shape == close.shape):
        raise ValueError("open/high/low/close exact paths must share a shape")
    rows, minutes = open_.shape
    if minutes != HORIZON_MINUTES:
        raise ValueError(f"full-horizon competing-risk paths must contain {HORIZON_MINUTES} minutes")
    stacked = np.stack(arrays, axis=2)
    if (
        not np.isfinite(stacked).all()
        or (stacked <= 0.0).any()
        or (high < low).any()
    ):
        raise ValueError("exact 1m OHLC paths must be finite, positive, and high >= low")
    if not isinstance(buffer_bps, (int, np.integer)) or int(buffer_bps) < 0:
        raise ValueError("buffer_bps must be a non-negative integer")

    atr = _finite_vector("oof_entry_atr_fraction", oof_entry_atr_fraction, rows, positive=True)
    cost = _finite_vector("execution_cost_return", execution_cost_return, rows)
    if (cost < 0.0).any():
        raise ValueError("execution_cost_return must be non-negative")
    entry = _finite_vector("execution_entry_price", execution_entry_price, rows, positive=True)
    sign = _finite_vector("side_sign", side_sign, rows)
    if not np.isin(sign, (-1.0, 1.0)).all():
        raise ValueError("side_sign must be canonical -1/+1")
    decision = pd.to_datetime(pd.Series(decision_utc), utc=True, errors="coerce")
    if decision.shape != (rows,) or decision.isna().any():
        raise ValueError("decision_utc must be one valid UTC timestamp per exact path")

    buffer_return = float(buffer_bps) / 10_000.0
    atr_upper = UPPER_ATR * atr
    floor_upper = np.full(rows, UPPER_RETURN_FLOOR)
    cost_upper = cost + buffer_return
    # Exact equality precedence is cost_plus_buffer, then ATR, then floor.
    # The number stays max(...); only the audit driver uses this tie rule.
    if use_upper_return_floor:
        upper = np.maximum.reduce((atr_upper, floor_upper, cost_upper))
        upper_driver = np.where(
            cost_upper >= np.maximum(atr_upper, floor_upper),
            "cost_plus_buffer",
            np.where(atr_upper >= floor_upper, "atr", "1p5pct_floor"),
        )
    else:
        upper = np.maximum(atr_upper, cost_upper)
        upper_driver = np.where(cost_upper >= atr_upper, "cost_plus_buffer", "atr")
    lower = LOWER_ATR * atr
    long_favorable = high / entry[:, None] - 1.0 >= upper[:, None]
    long_adverse = 1.0 - low / entry[:, None] >= lower[:, None]
    short_favorable = 1.0 - low / entry[:, None] >= upper[:, None]
    short_adverse = high / entry[:, None] - 1.0 >= lower[:, None]
    favorable_mask = np.where(sign[:, None] > 0.0, long_favorable, short_favorable)
    adverse_mask = np.where(sign[:, None] > 0.0, long_adverse, short_adverse)
    first_favorable = _first_index(favorable_mask)
    first_adverse = _first_index(adverse_mask)
    has_favorable = first_favorable >= 0
    has_adverse = first_adverse >= 0
    clean_favorable = has_favorable & (~has_adverse | (first_favorable < first_adverse))
    adverse_first = has_adverse & (~has_favorable | (first_adverse <= first_favorable))
    timeout = ~clean_favorable & ~adverse_first
    if not np.all(clean_favorable.astype(int) + adverse_first.astype(int) + timeout.astype(int) == 1):
        raise AssertionError("competing-risk labels must be exhaustive and mutually exclusive")
    same_minute_conflict = has_favorable & has_adverse & (first_favorable == first_adverse)

    endpoint_signed_return = np.where(
        sign > 0.0,
        close[:, -1] / entry - 1.0,
        1.0 - close[:, -1] / entry,
    )
    endpoint_favorable_margin = endpoint_signed_return - upper
    endpoint_adverse_margin = endpoint_signed_return + lower
    # Only a timeout has no observed ordering outcome.  Its terminal location
    # between the two barriers yields an explicitly *conditional* soft simplex:
    # at the adverse boundary (0,1,0), at midpoint (0,0,1), and at the
    # favourable boundary (1,0,0).  It is never substituted for a hit label.
    position = np.clip(endpoint_adverse_margin / (upper + lower), 0.0, 1.0)
    timeout_soft_clean = np.maximum(2.0 * position - 1.0, 0.0)
    timeout_soft_adverse = np.maximum(1.0 - 2.0 * position, 0.0)
    timeout_soft_timeout = 1.0 - timeout_soft_clean - timeout_soft_adverse
    if not np.allclose(
        timeout_soft_clean + timeout_soft_adverse + timeout_soft_timeout,
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("timeout-only soft viability simplex must sum to one")

    event = np.full(rows, "timeout", dtype=object)
    event[adverse_first] = "adverse_first"
    event[clean_favorable] = "clean_economic_favorable_first"
    first_event = np.where(clean_favorable, first_favorable, np.where(adverse_first, first_adverse, -1))
    first_fav_utc = decision + pd.to_timedelta(np.where(has_favorable, first_favorable, 0), unit="min")
    first_adv_utc = decision + pd.to_timedelta(np.where(has_adverse, first_adverse, 0), unit="min")
    first_event_utc = decision + pd.to_timedelta(np.where(first_event >= 0, first_event, 0), unit="min")
    label_end = decision + pd.Timedelta(minutes=HORIZON_MINUTES)
    return pd.DataFrame(
        {
            "cost_buffer_bps": np.full(rows, int(buffer_bps), dtype=np.int16),
            "cost_buffer_return": np.full(rows, buffer_return, dtype=np.float32),
            "oof_entry_atr_fraction": atr.astype(np.float32),
            "economic_upper_return": upper.astype(np.float32),
            "adverse_lower_return": lower.astype(np.float32),
            "upper_barrier_driver": upper_driver,
            "competing_risk_class": np.select(
                [timeout, adverse_first, clean_favorable], [0, 1, 2], default=-1
            ).astype(np.int8),
            "competing_risk_event": pd.array(event, dtype="string"),
            "timeout": timeout.astype(np.int8),
            "adverse_first": adverse_first.astype(np.int8),
            "clean_economic_favorable_first": clean_favorable.astype(np.int8),
            "same_minute_favorable_adverse_conflict": same_minute_conflict.astype(np.int8),
            "first_favorable_minute": np.where(has_favorable, first_favorable, np.nan).astype(np.float32),
            "first_adverse_minute": np.where(has_adverse, first_adverse, np.nan).astype(np.float32),
            "first_event_minute": np.where(first_event >= 0, first_event, np.nan).astype(np.float32),
            "first_favorable_utc": first_fav_utc.where(has_favorable, pd.NaT).to_numpy(),
            "first_adverse_utc": first_adv_utc.where(has_adverse, pd.NaT).to_numpy(),
            "first_event_utc": first_event_utc.where(first_event >= 0, pd.NaT).to_numpy(),
            "endpoint_signed_return": endpoint_signed_return.astype(np.float32),
            "endpoint_favorable_margin_return": endpoint_favorable_margin.astype(np.float32),
            "endpoint_adverse_margin_return": endpoint_adverse_margin.astype(np.float32),
            "timeout_soft_viability_valid": timeout.astype(np.int8),
            "timeout_soft_clean_economic_favorable_viability": np.where(timeout, timeout_soft_clean, np.nan).astype(np.float32),
            "timeout_soft_adverse_viability": np.where(timeout, timeout_soft_adverse, np.nan).astype(np.float32),
            "timeout_soft_timeout_viability": np.where(timeout, timeout_soft_timeout, np.nan).astype(np.float32),
            "label_resolution_utc": label_end.to_numpy(),
            "label_available_at": label_end.to_numpy(),
        }
    )


def _resolve_manifest_path(value: str, *, manifest_path: Path) -> Path:
    raw = Path(str(value))
    return raw if raw.is_absolute() else (ROOT / raw).resolve()


def _validate_source_manifest(labels_path: Path, manifest_path: Path) -> tuple[dict[str, Any], Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SOURCE_SCHEMA:
        raise ValueError("source policy-label manifest schema is invalid")
    output = manifest.get("output", {})
    if output.get("sha256") != _sha256(labels_path):
        raise ValueError("source policy-label manifest does not bind the labels parquet")
    source_path = _resolve_manifest_path(str(output.get("path", "")), manifest_path=manifest_path)
    if source_path != labels_path.resolve():
        raise ValueError("source policy-label manifest output path disagrees with --policy-labels")
    target_path = _resolve_manifest_path(str(manifest.get("source", {}).get("path_targets", "")), manifest_path=manifest_path)
    target_hash = manifest.get("source", {}).get("path_targets_sha256")
    if not target_path.exists() or target_hash != _sha256(target_path):
        raise ValueError("source policy-label manifest does not bind an intact PIT path-target source")
    if int(manifest.get("exit_policy_contract", {}).get("horizon_minutes", 0)) != HORIZON_MINUTES:
        raise ValueError("source policy labels do not bind the required 720-minute horizon")
    return manifest, target_path


def _load_source(policy_labels_path: Path, policy_manifest_path: Path) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    manifest, path_targets_path = _validate_source_manifest(policy_labels_path, policy_manifest_path)
    required_policy = [
        *IDENTITY,
        "execution_decision_utc",
        "execution_label_end_utc",
        "execution_entry_price",
        "execution_gross_ev_12h",
        "execution_cost_return",
        "execution_net_ev_12h",
        "execution_exit_hour",
    ]
    policy = pd.read_parquet(policy_labels_path, columns=required_policy)
    targets = pd.read_parquet(path_targets_path, columns=[*IDENTITY, "__path_auxiliary_atr_fraction__"])
    for name, frame in (("policy labels", policy), ("PIT path targets", targets)):
        if frame.duplicated(list(IDENTITY), keep=False).any():
            raise ValueError(f"{name} contains duplicate exact candidate identities")
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        if frame["__ts__"].isna().any():
            raise ValueError(f"{name} contains invalid UTC identity timestamps")
        frame["side_name"] = frame["side_name"].astype(str).str.lower()
        if not frame["side_name"].isin(("long", "short")).all():
            raise ValueError(f"{name} has non-canonical long/short identities")
    source = policy.merge(targets, on=list(IDENTITY), how="left", validate="one_to_one")
    if source["__path_auxiliary_atr_fraction__"].isna().any():
        raise ValueError("policy-label to PIT ATR exact identity join is incomplete")
    source["execution_decision_utc"] = pd.to_datetime(source["execution_decision_utc"], utc=True, errors="coerce")
    source["execution_label_end_utc"] = pd.to_datetime(source["execution_label_end_utc"], utc=True, errors="coerce")
    if source[["execution_decision_utc", "execution_label_end_utc"]].isna().any().any():
        raise ValueError("policy labels contain invalid execution decision/label-end UTC timestamps")
    expected_end = source["execution_decision_utc"] + pd.Timedelta(minutes=HORIZON_MINUTES)
    if not source["execution_label_end_utc"].eq(expected_end).all():
        raise ValueError("policy labels do not bind label end to decision + 720 minutes")
    for column, positive in (
        ("__path_auxiliary_atr_fraction__", True),
        ("execution_entry_price", True),
        ("execution_gross_ev_12h", False),
        ("execution_cost_return", False),
        ("execution_net_ev_12h", False),
        ("execution_exit_hour", False),
    ):
        values = pd.to_numeric(source[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (positive and (values <= 0.0).any()):
            raise ValueError(f"{column} must be finite" + (" and strictly positive" if positive else ""))
    if (source["execution_cost_return"].to_numpy(dtype=float) < 0.0).any():
        raise ValueError("execution_cost_return must be non-negative")
    if not np.allclose(
        source["execution_gross_ev_12h"].to_numpy(dtype=float) - source["execution_cost_return"].to_numpy(dtype=float),
        source["execution_net_ev_12h"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("source exact policy gross - cost does not equal net")
    source = source.rename(columns={"__path_auxiliary_atr_fraction__": "oof_entry_atr_fraction"})
    return source.sort_values(["execution_decision_utc", "__symbol__", "side_name", "candidate_id"], kind="stable").reset_index(drop=True), manifest, path_targets_path


def _complete_path_mask(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    good = (
        np.isfinite(values).all(axis=1)
        & (values > 0.0).all(axis=1)
        & (values[:, 1] >= values[:, 2])
    )
    prefix = np.concatenate(([0], np.cumsum(good, dtype=np.int64)))
    ends = offsets + HORIZON_MINUTES
    in_range = (offsets >= 0) & (ends <= len(good))
    complete = np.zeros(len(offsets), dtype=bool)
    complete[in_range] = prefix[ends[in_range]] - prefix[offsets[in_range]] == HORIZON_MINUTES
    return complete


def _materialize_symbol(
    source: pd.DataFrame,
    *,
    data_root: Path,
    buffers_bps: Sequence[int],
    batch_rows: int,
    use_upper_return_floor: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    symbol = str(source["__symbol__"].iloc[0])
    start = pd.Timestamp(source["execution_decision_utc"].min())
    end = pd.Timestamp(source["execution_decision_utc"].max()) + pd.Timedelta(minutes=HORIZON_MINUTES)
    bars = _load_symbol_bars(data_root, symbol, start, end)
    grid = pd.date_range(start, end, freq="min", inclusive="left", tz="UTC")
    dense = bars.reindex(grid).loc[:, list(PATH_COLUMNS)]
    values = dense.to_numpy(dtype=np.float64)
    offsets = ((source["execution_decision_utc"] - start) / pd.Timedelta(minutes=1)).astype(np.int64).to_numpy()
    complete = _complete_path_mask(values, offsets)
    if not complete.all():
        raise ValueError(f"immutable exact 1m path coverage is incomplete for {symbol}: {int((~complete).sum())} rows")
    parts: list[pd.DataFrame] = []
    for begin in range(0, len(source), batch_rows):
        stop = min(begin + batch_rows, len(source))
        batch = source.iloc[begin:stop].reset_index(drop=True)
        local_offsets = offsets[begin:stop]
        matrices = tuple(
            np.stack([values[offset : offset + HORIZON_MINUTES, column] for offset in local_offsets])
            for column in range(len(PATH_COLUMNS))
        )
        for buffer_bps in buffers_bps:
            labels = build_row_cost_aware_competing_risk_labels(
                *matrices,
                oof_entry_atr_fraction=batch["oof_entry_atr_fraction"].to_numpy(dtype=float),
                execution_cost_return=batch["execution_cost_return"].to_numpy(dtype=float),
                execution_entry_price=batch["execution_entry_price"].to_numpy(dtype=float),
                side_sign=np.where(batch["side_name"].eq("long"), 1.0, -1.0),
                decision_utc=batch["execution_decision_utc"],
                buffer_bps=int(buffer_bps),
                use_upper_return_floor=use_upper_return_floor,
            )
            # The signed source row already carries the PIT ATR fraction.
            # Retain one authoritative copy in the joined output.
            labels = labels.drop(columns=["oof_entry_atr_fraction"])
            parts.append(pd.concat([batch, labels], axis=1))
    return pd.concat(parts, ignore_index=True), {
        "__symbol__": symbol,
        "rows": int(len(source)),
        "complete_rows": int(complete.sum()),
        "coverage": float(complete.mean()),
    }


def _support_summary(labels: pd.DataFrame) -> pd.DataFrame:
    work = labels.copy()
    work["month"] = pd.to_datetime(work["__ts__"], utc=True).dt.strftime("%Y-%m")
    return (
        work.groupby(["cost_buffer_bps", "month", "side_name"], observed=True, sort=True)
        .agg(
            rows=("candidate_id", "size"),
            timeout_rate=("timeout", "mean"),
            adverse_first_rate=("adverse_first", "mean"),
            clean_economic_favorable_first_rate=("clean_economic_favorable_first", "mean"),
            mean_economic_upper_return=("economic_upper_return", "mean"),
            atr_upper_barrier_rate=("upper_barrier_driver", lambda values: float(values.eq("atr").mean())),
            floor_upper_barrier_rate=("upper_barrier_driver", lambda values: float(values.eq("1p5pct_floor").mean())),
            cost_plus_buffer_upper_barrier_rate=("upper_barrier_driver", lambda values: float(values.eq("cost_plus_buffer").mean())),
            same_minute_conflict_rows=("same_minute_favorable_adverse_conflict", "sum"),
        )
        .reset_index()
    )


def _buffer_change_summary(labels: pd.DataFrame) -> pd.DataFrame:
    """Make inert buffers and changed hard outcomes explicit.

    The primary grid always contains zero bps.  A custom grid which omits zero
    is still auditable: its smallest configured buffer is recorded as the
    reference rather than silently implying a 0-bps comparison.
    """

    buffers = sorted(int(value) for value in labels["cost_buffer_bps"].unique())
    reference = 0 if 0 in buffers else buffers[0]
    keys = list(IDENTITY)
    base = labels.loc[labels["cost_buffer_bps"].eq(reference), [
        *keys,
        "economic_upper_return",
        "upper_barrier_driver",
        "competing_risk_event",
    ]].rename(columns={
        "economic_upper_return": "reference_economic_upper_return",
        "upper_barrier_driver": "reference_upper_barrier_driver",
        "competing_risk_event": "reference_competing_risk_event",
    })
    summaries: list[dict[str, Any]] = []
    for buffer_bps in buffers:
        current = labels.loc[labels["cost_buffer_bps"].eq(buffer_bps), [
            *keys,
            "economic_upper_return",
            "upper_barrier_driver",
            "competing_risk_event",
        ]].merge(base, on=keys, how="inner", validate="one_to_one")
        if len(current) != len(base):
            raise AssertionError("every buffer must preserve the exact reference identity set")
        upper_changed = ~np.isclose(
            current["economic_upper_return"].to_numpy(dtype=float),
            current["reference_economic_upper_return"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
        driver_changed = current["upper_barrier_driver"].ne(
            current["reference_upper_barrier_driver"]
        ).to_numpy()
        label_changed = current["competing_risk_event"].ne(
            current["reference_competing_risk_event"]
        ).to_numpy()
        summaries.append({
            "reference_buffer_bps": reference,
            "cost_buffer_bps": int(buffer_bps),
            "rows": int(len(current)),
            "upper_return_changed_rows": int(upper_changed.sum()),
            "upper_driver_changed_rows": int(driver_changed.sum()),
            "competing_risk_label_changed_rows": int(label_changed.sum()),
            "upper_return_changed_rate": float(upper_changed.mean()),
            "upper_driver_changed_rate": float(driver_changed.mean()),
            "competing_risk_label_changed_rate": float(label_changed.mean()),
        })
    return pd.DataFrame(summaries)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    if int(args.batch_rows) < 1:
        raise ValueError("batch_rows must be positive")
    buffers = tuple(sorted(set(int(value) for value in args.buffer_bps)))
    if not buffers or any(value < 0 for value in buffers):
        raise ValueError("buffer_bps must contain one or more non-negative values")
    source, policy_manifest, path_targets_path = _load_source(args.policy_labels, args.policy_manifest)
    expected_store = _resolve_manifest_path(str(policy_manifest.get("store", {}).get("root", "")), manifest_path=args.policy_manifest)
    actual_store = canonical_kraken_execution_1m_root(args.data_root).resolve()
    if expected_store.resolve() != actual_store:
        raise ValueError("--data-root does not resolve to the immutable 1m store bound by the source manifest")

    stage = args.output_dir.parent / f".{args.output_dir.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    try:
        parts: list[pd.DataFrame] = []
        coverage_rows: list[dict[str, Any]] = []
        for _, indices in source.groupby("__symbol__", sort=True).indices.items():
            local, coverage = _materialize_symbol(
                source.iloc[np.asarray(indices, dtype=np.int64)].reset_index(drop=True),
                data_root=args.data_root,
                buffers_bps=buffers,
                batch_rows=int(args.batch_rows),
                use_upper_return_floor=not bool(args.omit_upper_return_floor),
            )
            parts.append(local)
            coverage_rows.append(coverage)
        result = pd.concat(parts, ignore_index=True).sort_values(
            [*IDENTITY, "cost_buffer_bps"], kind="stable"
        ).reset_index(drop=True)
        if not result.columns.is_unique:
            duplicates = result.columns[result.columns.duplicated()].tolist()
            raise AssertionError(f"output contains duplicate columns: {duplicates}")
        if len(result) != len(source) * len(buffers):
            raise AssertionError("output does not preserve every exact identity for every requested buffer")
        if result.duplicated([*IDENTITY, "cost_buffer_bps"], keep=False).any():
            raise AssertionError("output has duplicate exact identity/buffer pairs")
        class_sum = result[["timeout", "adverse_first", "clean_economic_favorable_first"]].sum(axis=1)
        if not class_sum.eq(1).all():
            raise AssertionError("hard competing-risk classes are not mutually exclusive")
        timeout = result["timeout"].eq(1)
        simplex = result.loc[timeout, [
            "timeout_soft_clean_economic_favorable_viability",
            "timeout_soft_adverse_viability",
            "timeout_soft_timeout_viability",
        ]].to_numpy(dtype=float)
        if len(simplex) and (not np.isfinite(simplex).all() or not np.allclose(simplex.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)):
            raise AssertionError("timeout-only soft viability distribution is invalid")
        if result.loc[~timeout, [
            "timeout_soft_clean_economic_favorable_viability",
            "timeout_soft_adverse_viability",
            "timeout_soft_timeout_viability",
        ]].notna().any().any():
            raise AssertionError("timeout-only soft viability must not be assigned to observed hit classes")
        expected_end = result["execution_decision_utc"] + pd.Timedelta(minutes=HORIZON_MINUTES)
        if not result["label_resolution_utc"].eq(expected_end).all() or not result["label_available_at"].eq(expected_end).all():
            raise AssertionError("label resolution/availability must be exact decision + 12h")
        if not np.allclose(
            result["execution_gross_ev_12h"].to_numpy(float) - result["execution_cost_return"].to_numpy(float),
            result["execution_net_ev_12h"].to_numpy(float), rtol=0.0, atol=1e-10,
        ):
            raise AssertionError("source gross/cost/net identity changed during materialization")

        labels_path = stage / "execution_ev_cost_aware_competing_risk_labels.parquet"
        coverage_path = stage / "coverage_by_symbol.csv"
        support_path = stage / "support_by_month_side_buffer.csv"
        buffer_changes_path = stage / "buffer_change_summary.csv"
        result.to_parquet(labels_path, index=False, compression="zstd")
        coverage = pd.DataFrame(coverage_rows).sort_values("__symbol__", kind="stable")
        coverage.to_csv(coverage_path, index=False)
        _support_summary(result).to_csv(support_path, index=False)
        buffer_changes = _buffer_change_summary(result)
        buffer_changes.to_csv(buffer_changes_path, index=False)
        manifest = {
            "schema": SCHEMA,
            "status": "completed_exact_1m_target_only_not_model_evidence",
            "rows": int(len(result)),
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "identity": list(IDENTITY),
            "identity_plus_buffer_unique": True,
            "lineage": {
                "policy_labels": {"path": str(args.policy_labels), "sha256": _sha256(args.policy_labels)},
                "policy_labels_manifest": {"path": str(args.policy_manifest), "sha256": _sha256(args.policy_manifest)},
                "pit_path_targets": {"path": str(path_targets_path), "sha256": _sha256(path_targets_path)},
                "immutable_execution_1m_root": str(actual_store),
            },
            "path_contract": {
                "cadence_minutes": 1,
                "path": "[execution_decision_utc, execution_decision_utc + 720m)",
                "path_minutes": HORIZON_MINUTES,
                "coverage": "fail_closed: every candidate requires all 720 finite positive OHLC minutes, high >= low; no fill/asof/interpolation",
                "actual_policy_exit": "ignored; labels always use full horizon",
            },
            "event_contract": {
                "orientation": "side-signed against frozen executable execution_entry_price",
                "upper_return": (
                    "max(1.5 * oof_entry_atr_fraction, execution_cost_return + cost_buffer_bps / 10000)"
                    if args.omit_upper_return_floor
                    else "max(1.5 * oof_entry_atr_fraction, 0.015, execution_cost_return + cost_buffer_bps / 10000)"
                ),
                "upper_return_floor_included": not bool(args.omit_upper_return_floor),
                "upper_barrier_driver": "atr | 1p5pct_floor | cost_plus_buffer; deterministic exact-tie precedence cost_plus_buffer, then atr, then 1p5pct_floor",
                "lower_return": "1.0 * oof_entry_atr_fraction",
                "buffers_bps": list(buffers),
                "classes": ["timeout", "adverse_first", "clean_economic_favorable_first"],
                "same_minute_favorable_adverse_tie": "adverse_first",
                "timeout_soft_viability": "timeout-only endpoint triangular simplex over adverse / timeout / clean-economic-favorable viability; never replaces an observed first-touch class",
                "label_resolution": "execution_decision_utc + 720m",
            },
            "accounting": {
                "source_identity_asserted": "execution_gross_ev_12h - execution_cost_return == execution_net_ev_12h",
                "cost_in_barrier": "row-specific source execution_cost_return; no second fee/spread subtraction",
            },
            "coverage": {
                "complete_rows": int(coverage["complete_rows"].sum()),
                "rows": int(coverage["rows"].sum()),
                "rate": float(coverage["complete_rows"].sum() / max(coverage["rows"].sum(), 1)),
            },
            "outputs": {
                "labels": {"path": str(args.output_dir / labels_path.name), "sha256": _sha256(labels_path), "rows": int(len(result))},
                "coverage_by_symbol": {"path": str(args.output_dir / coverage_path.name), "sha256": _sha256(coverage_path)},
                "support_by_month_side_buffer": {"path": str(args.output_dir / support_path.name), "sha256": _sha256(support_path)},
                "buffer_change_summary": {"path": str(args.output_dir / buffer_changes_path.name), "sha256": _sha256(buffer_changes_path)},
            },
        }
        _write_json(stage / "manifest.json", manifest)
        os.replace(stage, args.output_dir)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-labels", type=Path,
        default=ROOT / "data_perp/artifacts/execution_ev_policy_labels_12h_july20_20260726_v1/execution_ev_policy_labels.parquet",
    )
    parser.add_argument(
        "--policy-manifest", type=Path,
        default=ROOT / "data_perp/artifacts/execution_ev_policy_labels_12h_july20_20260726_v1/manifest.json",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data_perp")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--buffer-bps", type=int, action="append", default=[])
    parser.add_argument("--batch-rows", type=int, default=512)
    parser.add_argument(
        "--omit-upper-return-floor",
        action="store_true",
        help="supporting sensitivity only: use max(1.5 * ATR, row cost + buffer), without the 1.5%% floor",
    )
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    if not args.buffer_bps:
        args.buffer_bps = list(DEFAULT_BUFFERS_BPS)
    print(json.dumps(materialize(args), indent=2, default=str))
