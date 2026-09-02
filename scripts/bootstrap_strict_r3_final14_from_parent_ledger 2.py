#!/usr/bin/env python3
"""Export and bootstrap the immutable strict-R3 final-14 parent ledger.

This is a one-time, research/offline utility.  It consumes a canonical full-
history feature debug export plus the exact source-panel state, writes one
compact hash-bound parent ledger, replays the declared causal operators, and
atomically publishes a :class:`StrictR3Final14State` snapshot.  It never edits
an inference bundle or enables trading.

Semantic boundaries are explicit:

* raw OHLCV comes from the sealed point-in-time panel state;
* residual, simple-context, composite and OI parents come from the
  ``pre_causal_transform`` history;
* ``canonical_ret1h`` comes from ``post_causal_transform``;
* market-spectral sources come from the frozen ordered 97-row debug export.

The spectral and OI definition IDs are hashes of their real frozen contracts,
not caller-supplied labels.  Re-running against the same inputs is
deterministic, but the output directory is immutable and must not exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.inference.live_zscore_state import (  # noqa: E402
    RawRollingFeatureState,
    RollingZScoreState,
)
from extreme_price_movements.inference.price_memory_pipeline_state import (  # noqa: E402
    PriceMemoryPipelineState,
)
from extreme_price_movements.inference.residual_surprise_state import (  # noqa: E402
    OwnHistoryResidualState,
)
from extreme_price_movements.inference.simple_context_state import (  # noqa: E402
    SimpleContextFeatureState,
)
from extreme_price_movements.inference.spectral_oi_geometry_state import (  # noqa: E402
    SpectralOiGeometryState,
)
from extreme_price_movements.inference.strict_r3_final14_state import (  # noqa: E402
    FINAL14_FIELD_ORDER,
    StrictR3Final14State,
)
from scripts.update_strict_r3_feature_panel_state import STATE_SCHEMA  # noqa: E402


SCHEMA = "strict_r3_final14_parent_ledger_v1"
BOOTSTRAP_SCHEMA = "strict_r3_final14_bootstrap_v1"
RAW_PRICE_FIELDS = ("open", "high", "low", "close", "volume", "quote_volume")
RESIDUAL_PARENTS = (
    "excess_6h",
    "spike_score",
    "grind_score",
    "volume_price_corr_10h",
)
SIMPLE_CONTEXT_PARENTS = ("ret4h", "log_quote_volume", "ob_spread_bps")
COMPOSITE_PARENTS = (
    "bars_in_high_vol_state_log_norm",
    "ret48h_bench_resid",
)
POST_TRANSFORM_PARENTS = ("ret1h",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _history_paths(debug_dir: Path, stage: str, feature: str) -> tuple[Path, Path]:
    stem = hashlib.sha256(feature.encode("utf-8")).hexdigest()[:16]
    root = debug_dir / f"{stage}_history"
    return root / f"{stem}.parquet", root / f"{stem}.json"


def _read_history(
    debug_dir: Path,
    *,
    stage: str,
    feature: str,
    index: pd.DatetimeIndex,
    symbols: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    parquet, metadata_path = _history_paths(debug_dir, stage, feature)
    if not parquet.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"missing {stage} history for {feature}: {parquet} / {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "schema": "strict_r3_feature_parent_history_v1",
        "stage": stage,
        "feature": feature,
        "rows": len(index),
        "symbols": list(map(str, symbols)),
        "history_start": index[0].isoformat(),
        "history_end": index[-1].isoformat(),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"{stage}/{feature} metadata mismatch for {key}: "
                f"{metadata.get(key)!r} != {value!r}"
            )
    frame = pd.read_parquet(parquet)
    frame.index = pd.to_datetime(frame.index, utc=True)
    if not pd.DatetimeIndex(frame.index).equals(index):
        raise ValueError(f"{stage}/{feature} timestamp contract mismatch")
    if list(map(str, frame.columns)) != list(map(str, symbols)):
        raise ValueError(f"{stage}/{feature} symbol contract mismatch")
    values = frame.to_numpy(dtype=np.float32, copy=True)
    return values, {
        "parquet": str(parquet),
        "parquet_sha256": _sha256(parquet),
        "metadata": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
    }


def _read_spectral_tail(
    debug_dir: Path,
    *,
    selected_columns: Sequence[str],
    watermark: pd.Timestamp,
    lookback: int,
) -> tuple[pd.DatetimeIndex, np.ndarray, dict[str, Any]]:
    path = debug_dir / "market_spectral_source_tail.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen spectral source tail: {path}")
    long = pd.read_parquet(path)
    required = {"timestamp", "source_feature", "value"}
    if not required.issubset(long.columns):
        raise ValueError("spectral source tail has the wrong schema")
    long["timestamp"] = pd.to_datetime(long["timestamp"], utc=True)
    duplicate = long.duplicated(["timestamp", "source_feature"])
    if duplicate.any():
        raise ValueError("spectral source tail contains duplicate cells")
    observed_columns = set(long["source_feature"].astype(str).unique())
    expected_columns = set(map(str, selected_columns))
    if observed_columns != expected_columns:
        missing = sorted(expected_columns - observed_columns)
        unexpected = sorted(observed_columns - expected_columns)
        raise ValueError(
            "spectral source identity mismatch; "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    wide = long.pivot(index="timestamp", columns="source_feature", values="value")
    wide = wide.reindex(columns=list(selected_columns))
    if list(map(str, wide.columns)) != list(map(str, selected_columns)):
        raise ValueError("spectral source ordering mismatch")
    required_rows = 2 * int(lookback) + 1
    if len(wide) < required_rows:
        raise ValueError(
            f"spectral source has {len(wide)} rows; at least {required_rows} required"
        )
    wide = wide.tail(required_rows)
    index = pd.DatetimeIndex(wide.index)
    expected = pd.date_range(index[0], index[-1], freq="h", tz="UTC")
    if not index.equals(expected):
        raise ValueError("spectral source tail is not hourly contiguous")
    if index[-1] != watermark:
        raise ValueError("spectral source tail does not end at the panel watermark")
    return index, wide.to_numpy(dtype=np.float32, copy=True), {
        "path": str(path),
        "sha256": _sha256(path),
        "rows_available": int(long["timestamp"].nunique()),
        "rows_consumed": int(len(wide)),
    }


def _read_canonical_ret1h(
    debug_dir: Path,
    *,
    index: pd.DatetimeIndex,
    symbols: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read canonical ret1h at its direct-output/post-transform boundary.

    ``ret1h`` is produced by the canonical direct-output provider and is in
    the transform skip set.  A complete pre-transform history is consequently
    bit-identical to the post-transform history.  Prefer an explicit post
    history when present; otherwise require terminal pre/post snapshots to be
    bit-identical before accepting the complete direct-output history.
    """

    post_parquet, post_metadata = _history_paths(
        debug_dir, "post_causal_transform", "ret1h"
    )
    if post_parquet.is_file() and post_metadata.is_file():
        return _read_history(
            debug_dir,
            stage="post_causal_transform",
            feature="ret1h",
            index=index,
            symbols=symbols,
        )
    values, receipt = _read_history(
        debug_dir,
        stage="pre_causal_transform",
        feature="ret1h",
        index=index,
        symbols=symbols,
    )
    terminal: dict[str, np.ndarray] = {}
    terminal_sources: dict[str, str] = {}
    for stage in ("pre_causal_transform", "post_causal_transform"):
        path = debug_dir / f"{stage}_latest.parquet"
        if not path.is_file():
            raise FileNotFoundError(
                "canonical ret1h fallback requires both terminal stage snapshots"
            )
        frame = pd.read_parquet(path)
        subset = frame.loc[frame["feature"].astype(str).eq("ret1h")].copy()
        subset["timestamp"] = pd.to_datetime(subset["timestamp"], utc=True)
        if len(subset) != len(symbols) or subset["timestamp"].nunique() != 1:
            raise ValueError(f"{stage} terminal ret1h contract mismatch")
        if pd.Timestamp(subset["timestamp"].iloc[0]) != index[-1]:
            raise ValueError(f"{stage} terminal ret1h watermark mismatch")
        row = (
            subset.set_index(subset["symbol"].astype(str))["value"]
            .reindex(list(map(str, symbols)))
            .to_numpy(dtype=np.float32)
        )
        terminal[stage] = row
        terminal_sources[stage] = _sha256(path)
    if not np.array_equal(
        terminal["pre_causal_transform"],
        terminal["post_causal_transform"],
        equal_nan=True,
    ):
        raise ValueError("canonical ret1h changes across the transform boundary")
    receipt.update(
        {
            "semantic_stage": (
                "direct_output_pre_causal_transform; bit-identical post-transform "
                "because ret1h is in the canonical skip set"
            ),
            "terminal_stage_snapshot_sha256": terminal_sources,
            "terminal_pre_post_exact": True,
        }
    )
    return values, receipt


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_parent_ledger(
    *,
    panel_state_path: Path,
    debug_dir: Path,
    spectral_contract_path: Path,
    config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Export the exact semantic-stage parent arrays to one compact ledger."""

    panel_state = joblib.load(panel_state_path)
    if panel_state.get("schema") != STATE_SCHEMA:
        raise ValueError("unsupported strict-R3 source-panel state")
    symbols = tuple(map(str, panel_state.get("symbols", ())))
    panel = panel_state.get("panel")
    if not symbols or not isinstance(panel, Mapping):
        raise ValueError("source-panel state lacks its symbol/panel contract")
    close = panel.get("close")
    if not isinstance(close, pd.DataFrame) or close.empty:
        raise ValueError("source-panel state lacks close history")
    index = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True))
    expected_index = pd.date_range(index[0], index[-1], freq="h", tz="UTC")
    if not index.equals(expected_index):
        raise ValueError("source-panel history must be exactly hourly contiguous")
    if list(map(str, close.columns)) != list(symbols):
        raise ValueError("source-panel symbol order mismatch")

    spectral_contract = json.loads(spectral_contract_path.read_text())
    if spectral_contract.get("schema") != "strict_r3_market_spectral_source_state_v1":
        raise ValueError("unsupported frozen market-spectral source contract")
    selected_columns = tuple(map(str, spectral_contract.get("selected_columns", ())))
    if len(selected_columns) < 2 or len(set(selected_columns)) != len(selected_columns):
        raise ValueError("invalid frozen spectral source ordering")

    configured = dict(CFG)
    declared_oi_columns = tuple(
        map(
            str,
            (configured.get("MODEL_REGIME_COMPOSITE_EIGEN_GROUPS", {}) or {}).get(
                "open_interest", ()
            ),
        )
    )
    if len(declared_oi_columns) < 2 or len(set(declared_oi_columns)) != len(declared_oi_columns):
        raise ValueError("invalid frozen OI parent ordering")

    arrays: dict[str, np.ndarray] = {
        "timestamps_ns": index.asi8.astype(np.int64),
        "symbols": np.asarray(symbols, dtype=np.str_),
        "spectral_source_columns": np.asarray(selected_columns, dtype=np.str_),
    }
    sources: dict[str, Any] = {}
    for field in RAW_PRICE_FIELDS:
        frame = panel.get(field)
        if not isinstance(frame, pd.DataFrame):
            raise KeyError(f"source-panel state lacks raw field {field}")
        aligned = frame.reindex(index=index, columns=list(symbols))
        arrays[f"raw__{field}"] = aligned.to_numpy(dtype=np.float32, copy=True)

    for field in (*RESIDUAL_PARENTS, *SIMPLE_CONTEXT_PARENTS):
        values, receipt = _read_history(
            debug_dir,
            stage="pre_residual_consumers",
            feature=field,
            index=index,
            symbols=symbols,
        )
        arrays[f"pre__{field}"] = values
        sources[f"pre_residual_consumers/{field}"] = receipt
    breadth_ret4h, breadth_receipt = _read_history(
        debug_dir,
        stage="pre_direct_causal_output",
        feature="ret4h",
        index=index,
        symbols=symbols,
    )
    arrays["breadth__ret4h"] = breadth_ret4h
    sources["pre_direct_causal_output/ret4h"] = breadth_receipt
    for field in COMPOSITE_PARENTS:
        values, receipt = _read_history(
            debug_dir,
            stage="pre_direct_causal_output",
            feature=field,
            index=index,
            symbols=symbols,
        )
        arrays[f"pre__{field}"] = values
        sources[f"pre_direct_causal_output/{field}"] = receipt
    active_oi_columns: list[str] = []
    for field in declared_oi_columns:
        values, receipt = _read_history(
            debug_dir,
            stage="pre_direct_causal_output",
            feature=field,
            index=index,
            symbols=symbols,
        )
        if np.isfinite(values).any():
            active_oi_columns.append(field)
            arrays[f"pre__{field}"] = values
            sources[f"pre_direct_causal_output/{field}"] = receipt
    oi_columns = tuple(active_oi_columns)
    if len(oi_columns) < 2:
        raise ValueError("fewer than two active frozen OI parents")
    arrays["oi_parent_columns"] = np.asarray(oi_columns, dtype=np.str_)
    barrier_ret1h, barrier_receipt = _read_history(
        debug_dir,
        stage="pre_barrier_consumers",
        feature="ret1h",
        index=index,
        symbols=symbols,
    )
    arrays["barrier__ret1h"] = barrier_ret1h
    sources["pre_barrier_consumers/ret1h"] = barrier_receipt
    for field in POST_TRANSFORM_PARENTS:
        values, receipt = _read_canonical_ret1h(
            debug_dir, index=index, symbols=symbols
        )
        arrays[f"post__{field}"] = values
        sources[f"post_causal_transform/{field}"] = receipt

    lookback = int(configured.get("market_spectral_position_lookback", 48) or 48)
    spectral_index, spectral_source, spectral_receipt = _read_spectral_tail(
        debug_dir,
        selected_columns=selected_columns,
        watermark=index[-1],
        lookback=lookback,
    )
    arrays["spectral_timestamps_ns"] = spectral_index.asi8.astype(np.int64)
    arrays["spectral_source"] = spectral_source
    sources["market_spectral_source"] = spectral_receipt

    spectral_definition = {
        "schema": "strict_r3_spectral_definition_id_v1",
        "contract_sha256": _sha256(spectral_contract_path),
        "selected_columns": list(selected_columns),
        "source_keys": list(map(str, spectral_contract.get("source_keys", ()))),
        "lookback": lookback,
        "min_periods": int(configured.get("market_spectral_position_min_periods", 24) or 24),
        "top_k": int(configured.get("market_spectral_position_top_k", 3) or 3),
        "shrinkage": float(configured.get("market_spectral_position_shrinkage", 0.10) or 0.10),
    }
    oi_definition = {
        "schema": "strict_r3_oi_geometry_definition_id_v1",
        "group": "open_interest",
        "ordered_parents": list(oi_columns),
        "declared_parents": list(declared_oi_columns),
        "activation_rule": "retain declared parent iff finite anywhere in frozen bootstrap history",
        "config_sha256": _sha256(config_path),
    }
    arrays["spectral_definition_id"] = np.asarray(
        f"sha256:{_json_hash(spectral_definition)}"
    )
    arrays["oi_geometry_definition_id"] = np.asarray(
        f"sha256:{_json_hash(oi_definition)}"
    )
    _atomic_npz(output_path, arrays)
    return {
        "schema": SCHEMA,
        "ledger": str(output_path),
        "ledger_sha256": _sha256(output_path),
        "rows": int(len(index)),
        "symbols": int(len(symbols)),
        "history_start": index[0].isoformat(),
        "watermark": index[-1].isoformat(),
        "spectral_rows": int(len(spectral_index)),
        "spectral_required_rows": int(2 * lookback + 1),
        "spectral_definition_id": str(arrays["spectral_definition_id"].item()),
        "oi_geometry_definition_id": str(arrays["oi_geometry_definition_id"].item()),
        "panel_state": str(panel_state_path),
        "panel_state_sha256": _sha256(panel_state_path),
        "spectral_contract": str(spectral_contract_path),
        "spectral_contract_sha256": _sha256(spectral_contract_path),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "sources": sources,
        "outcome_columns_consumed": [],
    }


def _scalar(payload: Mapping[str, np.ndarray], name: str) -> str:
    value = np.asarray(payload[name])
    if value.size != 1:
        raise ValueError(f"ledger scalar {name} has the wrong shape")
    return str(value.item())


def bootstrap_state(
    *, ledger_path: Path,
    output_path: Path,
    working_dir: Path,
) -> tuple[StrictR3Final14State, dict[str, Any]]:
    """Replay the ledger and snapshot one composite final-14 state."""

    with np.load(ledger_path, allow_pickle=False) as loaded:
        payload = {key: loaded[key] for key in loaded.files}
    symbols = tuple(map(str, payload["symbols"].tolist()))
    index = pd.DatetimeIndex(
        pd.to_datetime(payload["timestamps_ns"].astype(np.int64), utc=True)
    )
    spectral_index = pd.DatetimeIndex(
        pd.to_datetime(payload["spectral_timestamps_ns"].astype(np.int64), utc=True)
    )
    spectral_columns = tuple(map(str, payload["spectral_source_columns"].tolist()))
    oi_columns = tuple(map(str, payload["oi_parent_columns"].tolist()))
    rows, width = len(index), len(symbols)
    expected = (rows, width)
    for name in RAW_PRICE_FIELDS:
        if payload[f"raw__{name}"].shape != expected:
            raise ValueError(f"raw ledger field {name} has the wrong shape")
    for name in (*RESIDUAL_PARENTS, *SIMPLE_CONTEXT_PARENTS, *COMPOSITE_PARENTS, *oi_columns):
        if payload[f"pre__{name}"].shape != expected:
            raise ValueError(f"pre-transform ledger field {name} has the wrong shape")
    if payload["post__ret1h"].shape != expected:
        raise ValueError("post-transform canonical ret1h has the wrong shape")
    if payload["barrier__ret1h"].shape != expected:
        raise ValueError("pre-direct barrier ret1h has the wrong shape")
    if payload["breadth__ret4h"].shape != expected:
        raise ValueError("post-portability breadth ret4h has the wrong shape")

    configured = dict(CFG)
    transform_window = 24 * 30
    winsor_qt = 0.02
    sigma_k = float(NormalDist().inv_cdf(1.0 - winsor_qt))
    price = PriceMemoryPipelineState(
        cache_dir=working_dir / "price_memory", symbols=symbols, atr_n=14
    )
    residual = OwnHistoryResidualState(symbols=symbols)
    simple = SimpleContextFeatureState(
        symbols=symbols, market_basket=list(configured.get("market_basket", ()) or ())
    )
    memory_transform = RollingZScoreState(
        ["memory_asymmetry_3ATR"],
        list(symbols),
        transform_window,
        sigma_k,
        winsor_qt=winsor_qt,
        buffer_dtype="float64",
    )
    high_vol_parent_transform = RollingZScoreState(
        ["bars_in_high_vol_state_log_norm"],
        list(symbols),
        transform_window,
        sigma_k,
        winsor_qt=winsor_qt,
    )
    pressure_robust = RawRollingFeatureState(
        op="robust_z",
        name="down_barrier_pressure_daily_donchian",
        symbols=list(symbols),
        window=transform_window,
    )
    pressure_dispersion = RawRollingFeatureState(
        op="std",
        name="down_barrier_pressure_daily_donchian::dispersion",
        symbols=list(symbols),
        window=transform_window,
    )
    terminal_price: dict[str, np.ndarray] | None = None
    terminal_residual: dict[str, np.ndarray] | None = None
    terminal_context: dict[str, np.ndarray] | None = None
    terminal_memory: np.ndarray | None = None
    terminal_high_vol_parent: np.ndarray | None = None
    terminal_pressure: np.ndarray | None = None
    terminal_pressure_std: np.ndarray | None = None

    # Replay price/residual/context and the two final causal transforms across
    # the complete canonical parent history.  Missing cells remain missing.
    for position, timestamp in enumerate(index):
        price_values = {
            name: payload[f"raw__{name}"][position] for name in RAW_PRICE_FIELDS
        }
        price_values["canonical_ret1h"] = payload["barrier__ret1h"][position]
        price_output = price.update(price_values, timestamp=timestamp)
        memory_output = memory_transform.update(
            {"memory_asymmetry_3ATR": price_output["memory_asymmetry_3ATR"]},
            timestamp=timestamp.isoformat(),
        )["memory_asymmetry_3ATR"]
        high_vol_output = high_vol_parent_transform.update(
            {
                "bars_in_high_vol_state_log_norm": payload[
                    "pre__bars_in_high_vol_state_log_norm"
                ][position]
            },
            timestamp=timestamp.isoformat(),
        )["bars_in_high_vol_state_log_norm"]
        pressure = price_output["down_barrier_pressure_daily_donchian"]
        pressure_output = pressure_robust.update(
            pressure, timestamp=timestamp.isoformat()
        )
        pressure_std = pressure_dispersion.update(
            pressure, timestamp=timestamp.isoformat()
        )
        residual_output = residual.update(
            {name: payload[f"pre__{name}"][position] for name in RESIDUAL_PARENTS},
            timestamp=timestamp,
        )
        context_output = simple.update(
            {
                **{
                    name: payload[f"pre__{name}"][position]
                    for name in SIMPLE_CONTEXT_PARENTS
                },
                "breadth_ret4h": payload["breadth__ret4h"][position],
            },
            timestamp=timestamp,
        )
        if position == rows - 1:
            terminal_price = price_output
            terminal_residual = residual_output
            terminal_context = context_output
            terminal_memory = memory_output
            terminal_high_vol_parent = high_vol_output
            terminal_pressure = pressure_output
            terminal_pressure_std = pressure_std

    lookback = int(configured.get("market_spectral_position_lookback", 48) or 48)
    spectral = SpectralOiGeometryState(
        symbols=symbols,
        spectral_source_columns=spectral_columns,
        oi_parent_columns=oi_columns,
        spectral_definition_id=_scalar(payload, "spectral_definition_id"),
        oi_geometry_definition_id=_scalar(payload, "oi_geometry_definition_id"),
        lookback=lookback,
        min_periods=int(configured.get("market_spectral_position_min_periods", 24) or 24),
        top_k=int(configured.get("market_spectral_position_top_k", 3) or 3),
        shrinkage=float(configured.get("market_spectral_position_shrinkage", 0.10) or 0.10),
    )
    spectral_positions = index.get_indexer(spectral_index)
    if (spectral_positions < 0).any():
        raise ValueError("spectral timestamps are outside the parent ledger")
    oi_tail = np.stack(
        [
            np.stack(
                [payload[f"pre__{name}"][position] for name in oi_columns], axis=0
            )
            for position in spectral_positions
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    spectral_outputs = spectral.bootstrap(
        timestamps=spectral_index,
        spectral_source=payload["spectral_source"],
        oi_parents=oi_tail,
    )

    state = StrictR3Final14State(
        symbols=symbols,
        price_memory=price,
        residual_surprise=residual,
        simple_context=simple,
        spectral_oi_geometry=spectral,
        transform_window=transform_window,
        winsor_qt=winsor_qt,
        memory_transform=memory_transform,
        high_vol_parent_transform=high_vol_parent_transform,
        pressure_robust=pressure_robust,
        pressure_dispersion=pressure_dispersion,
    )
    state.snapshot(output_path)
    restored = StrictR3Final14State.restore(output_path)
    if restored.contract_hash != state.contract_hash:
        raise AssertionError("restored final-14 contract hash changed")
    if restored.last_timestamp != index[-1].isoformat():
        raise AssertionError("restored final-14 watermark changed")

    if any(
        value is None
        for value in (
            terminal_price,
            terminal_residual,
            terminal_context,
            terminal_memory,
            terminal_high_vol_parent,
            terminal_pressure,
            terminal_pressure_std,
        )
    ):
        raise AssertionError("bootstrap did not retain terminal operator outputs")
    assert terminal_price is not None
    assert terminal_residual is not None
    assert terminal_context is not None
    assert terminal_memory is not None
    assert terminal_high_vol_parent is not None
    assert terminal_pressure is not None
    assert terminal_pressure_std is not None
    pressure_raw = np.asarray(
        terminal_price["down_barrier_pressure_daily_donchian"], dtype=np.float32
    )
    pressure_value = np.clip(
        np.asarray(terminal_pressure, dtype=np.float32), -8.0, 8.0
    )
    pressure_valid = (
        np.isfinite(pressure_raw)
        & (pressure_robust.count >= max(10, transform_window // 4))
        & np.isfinite(terminal_pressure_std)
        & (np.asarray(terminal_pressure_std) > np.float32(1e-8))
    )
    pressure_value[~pressure_valid] = np.nan
    donchian = np.asarray(
        terminal_price["bars_to_resistance_daily_donchian"], dtype=np.float32
    ).copy()
    vwap = np.asarray(
        terminal_price["bars_to_resistance_daily_vwap"], dtype=np.float32
    ).copy()
    for values in (donchian, vwap):
        finite = np.isfinite(values)
        values[finite] = np.clip(values[finite], -sigma_k, sigma_k)

    def quantile(values: np.ndarray, q: float) -> np.float32:
        finite = np.asarray(values, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        return np.float32(0.0 if not len(finite) else np.quantile(finite, q))

    composite_position = rows - 1
    high_vol = terminal_high_vol_parent
    ret48 = payload[f"pre__{COMPOSITE_PARENTS[1]}"][composite_position]
    terminal_values = {
        "negative_breadth_pct": terminal_context["negative_breadth_pct"],
        "down_barrier_pressure_daily_donchian": pressure_value,
        "q_upper_tail__bars_in_high_vol_state_log_norm": np.full(
            width, quantile(high_vol, 0.90), dtype=np.float32
        ),
        "state_spectral_eig_condition": spectral_outputs[
            "state_spectral_eig_condition"
        ][-1],
        "state_spectral_eig_gap_1_2": spectral_outputs[
            "state_spectral_eig_gap_1_2"
        ][-1],
        "state_spectral_eig_top3_share": spectral_outputs[
            "state_spectral_eig_top3_share"
        ][-1],
        "memory_asymmetry_3ATR": terminal_memory,
        "grind_score_surprise": terminal_residual["grind_score_surprise"],
        "bars_to_resistance_daily_donchian": donchian,
        "ret4h_peer_resid": terminal_context["ret4h_peer_resid"],
        "q_iqr__ret48h_bench_resid": np.full(
            width, quantile(ret48, 0.75) - quantile(ret48, 0.25), dtype=np.float32
        ),
        "spike_score_surprise": terminal_residual["spike_score_surprise"],
        "eig_effective_rank__open_interest": spectral_outputs[
            "eig_effective_rank__open_interest"
        ][-1],
        "bars_to_resistance_daily_vwap": vwap,
    }
    if tuple(terminal_values) != FINAL14_FIELD_ORDER:
        raise AssertionError("bootstrap terminal output order changed")
    terminal_path = output_path.parent / "terminal_final14_outputs.parquet"
    pd.DataFrame(terminal_values, index=pd.Index(symbols, name="symbol")).to_parquet(
        terminal_path, compression="zstd"
    )
    terminal_parent_path = (
        output_path.parent / "terminal_final14_raw_parent_audit.parquet"
    )
    pd.DataFrame(
        {
            "memory_asymmetry_3ATR": np.asarray(
                terminal_price["memory_asymmetry_3ATR"], dtype=np.float32
            ),
        },
        index=pd.Index(symbols, name="symbol"),
    ).to_parquet(terminal_parent_path, compression="zstd")
    manifest = {
        "schema": BOOTSTRAP_SCHEMA,
        "snapshot": str(output_path),
        "snapshot_sha256": _sha256(output_path),
        "ledger": str(ledger_path),
        "ledger_sha256": _sha256(ledger_path),
        "contract_hash": restored.contract_hash,
        "contract": restored.contract_payload(),
        "field_order": list(FINAL14_FIELD_ORDER),
        "rows_replayed": rows,
        "symbols": width,
        "history_start": index[0].isoformat(),
        "watermark": restored.last_timestamp,
        "spectral_rows_replayed": int(len(spectral_index)),
        "spectral_definition_id": spectral.spectral_definition_id,
        "oi_geometry_definition_id": spectral.oi_geometry_definition_id,
        "terminal_outputs": str(terminal_path),
        "terminal_outputs_sha256": _sha256(terminal_path),
        "terminal_raw_parent_audit": str(terminal_parent_path),
        "terminal_raw_parent_audit_sha256": _sha256(terminal_parent_path),
        "outcome_columns_consumed": [],
    }
    return restored, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-state", type=Path, required=True)
    parser.add_argument("--debug-snapshot-dir", type=Path, required=True)
    parser.add_argument("--spectral-contract", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "extreme_price_movements" / "config.py",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable bootstrap output exists: {args.out_dir}")
    staging = args.out_dir.with_name(args.out_dir.name + f".tmp.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"stale bootstrap staging directory exists: {staging}")
    staging.mkdir(parents=True)
    try:
        ledger_path = staging / "strict_r3_final14_parent_ledger.npz"
        ledger_manifest = export_parent_ledger(
            panel_state_path=args.panel_state,
            debug_dir=args.debug_snapshot_dir,
            spectral_contract_path=args.spectral_contract,
            config_path=args.config,
            output_path=ledger_path,
        )
        state_path = staging / "strict_r3_final14.state"
        _, bootstrap_manifest = bootstrap_state(
            ledger_path=ledger_path,
            output_path=state_path,
            working_dir=staging / ".bootstrap_work",
        )
        manifest = {
            "schema": "strict_r3_final14_parent_ledger_bootstrap_bundle_v1",
            "parent_ledger": ledger_manifest,
            "bootstrap": bootstrap_manifest,
        }
        manifest_path = staging / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        inventory = {
            path.relative_to(staging).as_posix(): _sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file() and ".bootstrap_work" not in path.parts
        }
        (staging / "artifact_inventory.json").write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n"
        )
        shutil.rmtree(staging / ".bootstrap_work", ignore_errors=True)
        staging.rename(args.out_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "contract_hash": bootstrap_manifest["contract_hash"],
        "watermark": bootstrap_manifest["watermark"],
        "rows_replayed": bootstrap_manifest["rows_replayed"],
        "spectral_rows_replayed": bootstrap_manifest["spectral_rows_replayed"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
