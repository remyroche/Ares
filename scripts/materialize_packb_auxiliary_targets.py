#!/usr/bin/env python3
"""Materialize five causal path-head targets for the canonical Pack-B top 40%.

Every canonical candidate is retained.  Target validity and its exact failure
reason are materialized instead of silently intersecting with an older label
population. ATR normalization uses causal raw-price Wilder ATR at the canonical
signal timestamp; the first future path bar begins one hour later at the
executable decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_candidate_population import (  # noqa: E402
    candidate_identity_sha256,
)
from extreme_price_movements.data_store import PartitionedOHLCVStore  # noqa: E402
from extreme_price_movements.path_auxiliary_targets import (  # noqa: E402
    ALL_SUPPORTIVE_LABEL_COLUMNS,
    TARGET_COLUMNS,
)
from extreme_price_movements.path_auxiliary_targets import (
    TARGET_SCHEMA as TARGET_KERNEL_SCHEMA,
)
from extreme_price_movements.training_resource_guard import (  # noqa: E402
    TrainingResourceGuard,
    TrainingResourceLimits,
)
from scripts.materialize_path_auxiliary_targets import (  # noqa: E402
    LABEL_RESOLUTION_COLUMN,
    OUTPUT_TARGET_COLUMNS,
    SymbolBars,
    materialize_batch_targets,
)
from scripts.run_packb_pre_march_side_ae import _git_revision  # noqa: E402

SCHEMA = "packb_canonical_path_auxiliary_targets_v1"
TARGET_SCHEMA = "packb_path_auxiliary_targets_v7_signal_atr_causal"
DEFAULT_TOP40 = (
    ROOT / "data_perp/artifacts/packb_side_local_top40_20260724_v1_31_8/"
    "base_candidate_population.parquet"
)
DEFAULT_TOP40_MANIFEST = DEFAULT_TOP40.with_name("manifest.json")
DEFAULT_OHLCV_ROOT = ROOT / "data_perp/exchanges/krakenfutures"
DEFAULT_OUTPUT = (
    ROOT / "data_perp/artifacts/packb_path_auxiliary_targets_20260725_v1_31_8"
)
ATR_COLUMN = "__path_auxiliary_atr_fraction__"
ATR_SOURCE_COLUMN = "raw_wilder_atr14_fraction_at_signal"
INVALID_REASON_COLUMN = "__path_auxiliary_invalid_reason__"
IDENTITY_COLUMNS = ("__ts__", "__symbol__", "side_name", "candidate_id")
MIN_VALID_RATE_PER_SIDE = 0.95
ATR_PERIOD = 14
ATR_BURN_IN_DAYS = 90


class PackBAuxiliaryTargetError(RuntimeError):
    """Raised when exact canonical auxiliary-target lineage cannot be proven."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def wilder_atr_fraction(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    period: int = ATR_PERIOD,
) -> np.ndarray:
    """Return causal raw-price Wilder ATR divided by contemporaneous close."""

    high_values = np.asarray(high, dtype=np.float64)
    low_values = np.asarray(low, dtype=np.float64)
    close_values = np.asarray(close, dtype=np.float64)
    if (
        high_values.ndim != 1
        or high_values.shape != low_values.shape
        or high_values.shape != close_values.shape
        or int(period) < 2
    ):
        raise PackBAuxiliaryTargetError("invalid Wilder ATR inputs")
    tr = np.full(len(close_values), np.nan, dtype=np.float64)
    if len(tr):
        tr[0] = high_values[0] - low_values[0]
    if len(tr) > 1:
        previous_close = close_values[:-1]
        tr[1:] = np.maximum(
            high_values[1:] - low_values[1:],
            np.maximum(
                np.abs(high_values[1:] - previous_close),
                np.abs(low_values[1:] - previous_close),
            ),
        )
    atr = (
        pd.Series(tr)
        .ewm(alpha=1.0 / float(period), adjust=False, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )
    return np.divide(
        atr,
        close_values,
        out=np.full(len(close_values), np.nan, dtype=np.float64),
        where=np.isfinite(close_values) & (close_values > 0.0),
    ).astype(np.float32)


def load_bars_and_signal_atr(
    root: Path,
    symbols: list[str],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    decision_delay_hours: int,
    horizon_hours: int,
    atr_period: int = ATR_PERIOD,
    atr_burn_in_days: int = ATR_BURN_IN_DAYS,
    resource_guard: TrainingResourceGuard | None = None,
) -> tuple[dict[str, SymbolBars], dict[str, np.ndarray]]:
    """Load bounded OHLCV once and derive causal signal-time ATR per symbol."""

    store = PartitionedOHLCVStore(root_dir=str(root), timeframe="1h")
    read_start = start - pd.Timedelta(days=int(atr_burn_in_days))
    read_end = end + pd.Timedelta(hours=int(decision_delay_hours) + int(horizon_hours))
    bars_by_symbol: dict[str, SymbolBars] = {}
    atr_by_symbol: dict[str, np.ndarray] = {}
    for position, symbol in enumerate(symbols, start=1):
        frame = store.load(
            symbol,
            columns=["open", "high", "low", "close"],
            start_ts=read_start,
            end_ts=read_end,
        )
        if frame.empty:
            continue
        frame = frame.sort_index()
        index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        finite_index = ~index.isna()
        if not np.any(finite_index):
            continue
        index_ns = index[finite_index].astype("int64").to_numpy()
        open_values = pd.to_numeric(
            frame.loc[finite_index, "open"], errors="coerce"
        ).to_numpy(np.float64)
        high_values = pd.to_numeric(
            frame.loc[finite_index, "high"], errors="coerce"
        ).to_numpy(np.float64)
        low_values = pd.to_numeric(
            frame.loc[finite_index, "low"], errors="coerce"
        ).to_numpy(np.float64)
        close_values = pd.to_numeric(
            frame.loc[finite_index, "close"], errors="coerce"
        ).to_numpy(np.float64)
        bars_by_symbol[symbol] = SymbolBars(
            index_ns=index_ns,
            open=open_values,
            high=high_values,
            low=low_values,
        )
        atr_by_symbol[symbol] = wilder_atr_fraction(
            high_values, low_values, close_values, period=atr_period
        )
        if resource_guard is not None and (
            position == 1 or position % 25 == 0 or position == len(symbols)
        ):
            resource_guard.checkpoint(f"packb_auxiliary_targets:load_symbol_{position}")
    return bars_by_symbol, atr_by_symbol


def align_signal_atr(
    population: pd.DataFrame,
    bars_by_symbol: Mapping[str, SymbolBars],
    atr_by_symbol: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Align ATR at the canonical signal timestamp without as-of fallback."""

    timestamps = pd.to_datetime(population["__ts__"], utc=True, errors="coerce")
    symbols = population["__symbol__"].astype(str).to_numpy()
    signal_ns = timestamps.astype("int64").to_numpy()
    output = np.full(len(population), np.nan, dtype=np.float32)
    for symbol in pd.unique(symbols):
        rows = np.flatnonzero(symbols == symbol)
        bars = bars_by_symbol.get(str(symbol))
        atr = atr_by_symbol.get(str(symbol))
        if bars is None or atr is None or len(atr) != len(bars.index_ns):
            continue
        positions = np.searchsorted(bars.index_ns, signal_ns[rows])
        bounded = np.minimum(positions, len(bars.index_ns) - 1)
        exact = (positions < len(bars.index_ns)) & (
            bars.index_ns[bounded] == signal_ns[rows]
        )
        output[rows[exact]] = np.asarray(atr, dtype=np.float32)[positions[exact]]
    return output


def derive_invalid_reasons(
    frame: pd.DataFrame,
    bars_by_symbol: Mapping[str, SymbolBars],
    *,
    decision_delay_hours: int = 1,
    horizon_hours: int = 12,
) -> np.ndarray:
    """Classify target availability one-for-one without dropping any row."""

    required = {"__ts__", "__symbol__", "side_name", ATR_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise PackBAuxiliaryTargetError(
            f"cannot derive target validity; missing columns: {missing}"
        )
    rows = len(frame)
    reasons = np.full(rows, "valid", dtype=object)
    timestamps = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    symbols = frame["__symbol__"].astype(str).to_numpy()
    sides = frame["side_name"].astype(str).str.lower().to_numpy()
    atr = pd.to_numeric(frame[ATR_COLUMN], errors="coerce").to_numpy(np.float64)
    bad_identity = timestamps.isna().to_numpy() | ~np.isin(sides, ("long", "short"))
    reasons[bad_identity] = "invalid_identity_or_side"
    bad_atr = ~np.isfinite(atr) | (atr <= 0.0)
    reasons[(reasons == "valid") & bad_atr] = "missing_or_nonpositive_signal_atr"
    signal_ns = timestamps.astype("int64").to_numpy()
    decision_ns = signal_ns + int(pd.Timedelta(hours=decision_delay_hours).value)
    hour_ns = int(pd.Timedelta(hours=1).value)
    horizon = int(horizon_hours)
    for symbol in pd.unique(symbols):
        selected = np.flatnonzero((symbols == symbol) & (reasons == "valid"))
        if not len(selected):
            continue
        bars = bars_by_symbol.get(str(symbol))
        if bars is None or not len(bars.index_ns):
            reasons[selected] = "missing_symbol_ohlcv"
            continue
        positions = np.searchsorted(bars.index_ns, decision_ns[selected])
        bounded = np.minimum(positions, len(bars.index_ns) - 1)
        exact = (positions < len(bars.index_ns)) & (
            bars.index_ns[bounded] == decision_ns[selected]
        )
        reasons[selected[~exact]] = "missing_exact_decision_bar"
        exact_rows = selected[exact]
        exact_positions = positions[exact]
        complete = (exact_positions + horizon) <= len(bars.index_ns)
        reasons[exact_rows[~complete]] = "incomplete_12h_path"
        path_rows = exact_rows[complete]
        path_positions = exact_positions[complete]
        if not len(path_rows):
            continue
        if horizon > 1:
            contiguous_windows = np.lib.stride_tricks.sliding_window_view(
                np.diff(bars.index_ns) == hour_ns, horizon - 1
            )[path_positions]
            contiguous = np.all(contiguous_windows, axis=1)
        else:
            contiguous = np.ones(len(path_rows), dtype=bool)
        reasons[path_rows[~contiguous]] = "noncontiguous_12h_path"
        finite_rows = path_rows[contiguous]
        finite_positions = path_positions[contiguous]
        if not len(finite_rows):
            continue
        high_windows = np.lib.stride_tricks.sliding_window_view(bars.high, horizon)[
            finite_positions
        ]
        low_windows = np.lib.stride_tricks.sliding_window_view(bars.low, horizon)[
            finite_positions
        ]
        finite = (
            np.isfinite(bars.open[finite_positions])
            & (bars.open[finite_positions] > 0.0)
            & np.all(np.isfinite(high_windows), axis=1)
            & np.all(np.isfinite(low_windows), axis=1)
            & np.all(high_windows > 0.0, axis=1)
            & np.all(low_windows > 0.0, axis=1)
        )
        reasons[finite_rows[~finite]] = "nonfinite_12h_ohlcv"
    return reasons.astype(str)


def build_target_frame(
    population: pd.DataFrame,
    atr_values: np.ndarray,
    bars_by_symbol: Mapping[str, SymbolBars],
    *,
    decision_delay_hours: int = 1,
    horizon_hours: int = 12,
) -> pd.DataFrame:
    """Return the exact population with target columns and explicit attrition."""

    required = {
        *IDENTITY_COLUMNS,
        "oos_fold",
        "selected_top40",
        "prediction_source",
    }
    missing = sorted(required.difference(population.columns))
    if missing:
        raise PackBAuxiliaryTargetError(f"top40 population misses columns: {missing}")
    if len(atr_values) != len(population):
        raise PackBAuxiliaryTargetError("ATR values are not aligned to the population")
    frame = population.loc[
        :,
        [
            *IDENTITY_COLUMNS,
            "oos_fold",
            "selected_top40",
            "prediction_source",
        ],
    ].copy()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    frame[ATR_COLUMN] = np.asarray(atr_values, dtype=np.float32)
    reasons = derive_invalid_reasons(
        frame,
        bars_by_symbol,
        decision_delay_hours=decision_delay_hours,
        horizon_hours=horizon_hours,
    )
    targets = materialize_batch_targets(
        frame,
        dict(bars_by_symbol),
        decision_delay_hours=decision_delay_hours,
        horizon_hours=horizon_hours,
    )
    for column, values in targets.items():
        frame[column] = values
    valid = frame["__path_auxiliary_target_valid__"].to_numpy(dtype=bool)
    if not np.array_equal(valid, reasons == "valid"):
        mismatch = int(np.sum(valid != (reasons == "valid")))
        raise PackBAuxiliaryTargetError(
            f"target builder and explicit invalid reasons disagree on {mismatch} rows"
        )
    frame.loc[~valid, LABEL_RESOLUTION_COLUMN] = pd.NaT
    frame[INVALID_REASON_COLUMN] = reasons
    frame["__bars_to_adverse_extreme_before_mfe_12h__"] = frame[
        "__bars_before_price_stops_decreasing_12h__"
    ]
    frame["__path_auxiliary_atr_source__"] = ATR_SOURCE_COLUMN
    frame["__path_auxiliary_atr_available_at__"] = frame["__ts__"]
    if (
        len(frame) != len(population)
        or frame["candidate_id"].duplicated().any()
        or not frame["selected_top40"].astype(bool).all()
        or set(frame["prediction_source"].astype(str)) != {"outer_oof_fold_model"}
    ):
        raise PackBAuxiliaryTargetError(
            "target frame does not preserve the exact canonical population"
        )
    return frame.sort_values(
        ["__ts__", "__symbol__", "side_name"], kind="mergesort"
    ).reset_index(drop=True)


def _group_counts(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    counts = (
        frame.groupby(columns, observed=True, dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    return counts.to_dict(orient="records")


def _target_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    valid = frame["__path_auxiliary_target_valid__"].eq(1)
    for side in ("long", "short"):
        side_rows = frame.loc[frame["side_name"].eq(side)]
        side_valid = side_rows.loc[valid.loc[side_rows.index]]
        output[side] = {
            "rows": len(side_rows),
            "valid_rows": len(side_valid),
            "valid_rate": len(side_valid) / max(len(side_rows), 1),
            "meaningful_mfe_reached_rate": float(
                side_valid["__meaningful_mfe_reached_12h__"].mean()
            )
            if len(side_valid)
            else None,
            "mixture_diagnostics": {
                "peak_zero_rate": float(
                    side_valid["__peak_mfe_atr_12h__"].eq(0.0).mean()
                )
                if len(side_valid)
                else None,
                "peak_clip_10atr_rate": float(
                    side_valid["__peak_mfe_atr_12h__"].ge(10.0).mean()
                )
                if len(side_valid)
                else None,
                "timing_right_censored_12h_rate": float(
                    side_valid["__meaningful_mfe_reached_12h__"].eq(0).mean()
                )
                if len(side_valid)
                else None,
                "mae_clip_10atr_rate": float(
                    side_valid["__mae_before_meaningful_mfe_atr_12h__"].ge(10.0).mean()
                )
                if len(side_valid)
                else None,
                "future_slope_clip_10atr_per_hour_rate": float(
                    side_valid["__future_slope_atr_per_hour_12h__"].ge(10.0).mean()
                )
                if len(side_valid)
                else None,
            },
            "raw_targets": {
                column: {
                    "mean": float(side_valid[column].mean()),
                    "std": float(side_valid[column].std()),
                    "p50": float(side_valid[column].quantile(0.50)),
                    "p90": float(side_valid[column].quantile(0.90)),
                }
                for column in (
                    "__peak_mfe_atr_12h__",
                    "__time_to_first_meaningful_mfe_hours_12h__",
                    "__mae_before_meaningful_mfe_atr_12h__",
                    "__bars_before_price_stops_decreasing_12h__",
                    "__future_slope_atr_per_hour_12h__",
                )
            },
        }
    return output


def run(
    *,
    top40_path: Path,
    top40_manifest_path: Path,
    ohlcv_root: Path,
    destination: Path,
    decision_delay_hours: int = 1,
    horizon_hours: int = 12,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite auxiliary targets: {destination}")
    top40_manifest = json.loads(top40_manifest_path.read_text(encoding="utf-8"))
    if (
        top40_manifest.get("output", {}).get("sha256") != _sha256(top40_path)
        or top40_manifest.get("selected_rows") != 300315
        or top40_manifest.get("source_rows") != 744251
    ):
        raise PackBAuxiliaryTargetError("canonical top40 source binding changed")
    revision = _git_revision()
    population = pd.read_parquet(top40_path)
    if (
        len(population) != 300315
        or population["candidate_id"].astype(str).duplicated().any()
    ):
        raise PackBAuxiliaryTargetError("canonical top40 population identity changed")
    stage = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    try:
        stage.mkdir(parents=True)
        guard = TrainingResourceGuard(
            limits=TrainingResourceLimits(check_interval_seconds=0.0),
            disk_path=destination.parent,
            telemetry_path=stage / "training_resource_telemetry.jsonl",
        )
        guard.preflight("packb_auxiliary_targets:start")
        timestamps = pd.to_datetime(population["__ts__"], utc=True, errors="raise")
        bars, atr_by_symbol = load_bars_and_signal_atr(
            ohlcv_root,
            sorted(population["__symbol__"].astype(str).unique()),
            start=timestamps.min(),
            end=timestamps.max(),
            decision_delay_hours=decision_delay_hours,
            horizon_hours=horizon_hours,
            resource_guard=guard,
        )
        atr = align_signal_atr(population, bars, atr_by_symbol)
        del atr_by_symbol
        guard.checkpoint("packb_auxiliary_targets:ohlcv_loaded")
        targets = build_target_frame(
            population,
            atr,
            bars,
            decision_delay_hours=decision_delay_hours,
            horizon_hours=horizon_hours,
        )
        del bars
        guard.checkpoint("packb_auxiliary_targets:targets_built")
        statistics = _target_statistics(targets)
        failed_sides = [
            side
            for side, values in statistics.items()
            if float(values["valid_rate"]) < MIN_VALID_RATE_PER_SIDE
        ]
        if failed_sides:
            raise PackBAuxiliaryTargetError(
                "path validity is below the production floor for: "
                + ", ".join(failed_sides)
            )
        output_path = stage / "targets.parquet"
        targets.to_parquet(
            output_path, index=False, compression="zstd", compression_level=5
        )
        result = {
            "schema": SCHEMA,
            "target_schema": TARGET_SCHEMA,
            "target_kernel_schema": TARGET_KERNEL_SCHEMA,
            "status": "MATERIALIZED_EXACT_CANONICAL_FIVE_HEAD_TARGETS",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": revision,
            "population": {
                "path": str(top40_path),
                "sha256": _sha256(top40_path),
                "manifest_sha256": _sha256(top40_manifest_path),
                "rows": len(population),
                "candidate_identity_sha256": candidate_identity_sha256(
                    population, columns=IDENTITY_COLUMNS
                ),
            },
            "atr_contract": {
                "source": ATR_SOURCE_COLUMN,
                "availability": "signal timestamp; strictly before +1h decision",
                "semantic_change": (
                    "new causal target geometry; not a bitwise regeneration of "
                    "historical decision-bar adaptive ATR labels"
                ),
                "formula": (
                    "raw-price true range; Wilder EWM alpha=1/14; divided by "
                    "signal-bar close"
                ),
                "join": "exact OHLCV __symbol__ + signal __ts__; no as-of or fill",
                "period": ATR_PERIOD,
                "burn_in_days": ATR_BURN_IN_DAYS,
                "ohlcv_root": str(ohlcv_root),
            },
            "path_contract": {
                "ohlcv_root": str(ohlcv_root),
                "decision_timestamp": "__ts__ + 1h",
                "first_path_bar": "open/high/low at decision timestamp",
                "horizon_hours": horizon_hours,
                "label_resolution": "__ts__ + 13h",
                "complete_contiguous_hourly_bars_required": True,
                "signal_clock_authority": (
                    "canonical top40 __ts__; legacy raw-label __ts__ is not consumed"
                ),
            },
            "outputs": {
                "path": str(destination / output_path.name),
                "sha256": _sha256(output_path),
                "rows": len(targets),
                "columns": len(targets.columns),
                "candidate_identity_sha256": candidate_identity_sha256(
                    targets, columns=IDENTITY_COLUMNS
                ),
            },
            "target_columns": dict(TARGET_COLUMNS),
            "supportive_label_columns": list(ALL_SUPPORTIVE_LABEL_COLUMNS),
            "all_materialized_target_columns": list(OUTPUT_TARGET_COLUMNS),
            "semantic_correction": {
                "legacy_head_name": "bars_before_price_stops_decreasing",
                "actual_primary_semantics": (
                    "one-based bar of the adverse extreme before meaningful MFE"
                ),
                "explicit_alias": "__bars_to_adverse_extreme_before_mfe_12h__",
                "confirmed_reversal_support_target": (
                    "__bars_to_confirmed_adverse_trough__"
                ),
            },
            "recommended_model_contract": {
                "peak_mfe_12h_atr": (
                    "hurdle probability plus conditional magnitude/upper quantile"
                ),
                "time_to_first_meaningful_mfe": (
                    "discrete survival or event probability plus conditional time"
                ),
                "mae_before_meaningful_mfe_atr": (
                    "hit probability plus separate conditional hit/no-hit risk"
                ),
                "bars_before_price_stops_decreasing": (
                    "benchmark legacy adverse-extreme target against the confirmed "
                    "two-bar adverse-trough support target"
                ),
                "future_slope_atr_per_hour": (
                    "retain only with incremental OOF economic value beyond peak/time"
                ),
            },
            "retention_contract": {
                "all_population_rows_retained": len(targets) == len(population),
                "invalid_reason_column": INVALID_REASON_COLUMN,
                "minimum_valid_rate_per_side": MIN_VALID_RATE_PER_SIDE,
            },
            "statistics": statistics,
            "attrition_by_side_fold_reason": _group_counts(
                targets, ["side_name", "oos_fold", INVALID_REASON_COLUMN]
            ),
        }
        _write_json(stage / "manifest.json", result)
        guard.checkpoint("packb_auxiliary_targets:complete")
        os.replace(stage, destination)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top40", type=Path, default=DEFAULT_TOP40)
    parser.add_argument("--top40-manifest", type=Path, default=DEFAULT_TOP40_MANIFEST)
    parser.add_argument("--ohlcv-root", type=Path, default=DEFAULT_OHLCV_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--decision-delay-hours", type=int, default=1)
    parser.add_argument("--horizon-hours", type=int, default=12)
    args = parser.parse_args()
    result = run(
        top40_path=args.top40,
        top40_manifest_path=args.top40_manifest,
        ohlcv_root=args.ohlcv_root,
        destination=args.output_dir,
        decision_delay_hours=args.decision_delay_hours,
        horizon_hours=args.horizon_hours,
    )
    print(json.dumps(_jsonable(result), sort_keys=True))


if __name__ == "__main__":
    main()
