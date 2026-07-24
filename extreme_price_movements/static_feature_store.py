"""Shared static-feature compute, state, and parity-block contract.

This module deliberately stops before fitted model transforms.  In particular,
AE/GMM assignment belongs to the frozen model artifact and must never be
recomputed as part of raw feature materialization.  Training and inference use
the same ``compute_static_features`` entry point for causal OHLCV/OI/funding/
order-book/market feature formulas, while keeping their separate input windows
and output stores explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


STATIC_FEATURE_ENDPOINT_VERSION = "static_feature_endpoint_v1"
STATIC_FEATURE_BLOCK_COLUMNS_PER_FILE = 32
_MODEL_DERIVED_PREFIXES = ("aegmm_", "gmm_", "ae_")


def resolve_static_feature_save_workers(
    cfg: Mapping[str, Any] | None = None,
    *,
    default: int = 4,
) -> int:
    """Return one bounded persistence-worker contract for every caller."""

    configured = os.getenv("EPM_STATIC_FEATURE_SAVE_WORKERS")
    if configured is None and cfg is not None:
        configured = cfg.get("feature_save_workers")
    try:
        workers = int(configured if configured is not None else default)
    except (TypeError, ValueError):
        workers = int(default)
    return max(2, min(workers, 32))


@dataclass(frozen=True)
class StaticFeatureResult:
    """Raw, causal features emitted by the common static feature endpoint."""

    features: dict[str, Any]
    index: pd.Index
    columns: list[str]
    runtime_cfg: dict[str, Any]


@dataclass(frozen=True)
class StaticMarketContext:
    """Causal market-wide context used by the same static endpoint."""

    market_features: pd.DataFrame
    regime_gates: pd.DataFrame
    used_state: bool = False
    state_path: str | None = None


_MARKET_AGGREGATE_STATE_VERSION = 1
_MARKET_AGGREGATE_FIELDS = ("close", "high", "low", "volume")


@dataclass
class MarketAggregateState:
    """Persisted NumPy history for exact market-transform incremental updates.

    The state stores only four cross-sectional aggregate input series.  Keeping
    the inputs, rather than derived gates, means a repaired trailing candle is
    recalculated by the same Numba market formulas as a full feature run.
    """

    frame: pd.DataFrame
    basket_hash: str

    @classmethod
    def load(cls, path: str | Path, *, basket_hash: str) -> "MarketAggregateState | None":
        target = Path(path)
        if not target.exists():
            return None
        try:
            with np.load(target, allow_pickle=False) as data:
                meta = json.loads(str(data["metadata"].item()))
                if int(meta.get("version", -1)) != _MARKET_AGGREGATE_STATE_VERSION:
                    return None
                if str(meta.get("basket_hash", "")) != str(basket_hash):
                    return None
                fields = [str(value) for value in meta.get("fields", [])]
                if fields != list(_MARKET_AGGREGATE_FIELDS):
                    return None
                index = pd.to_datetime(
                    np.asarray(data["index_ns"], dtype=np.int64), utc=True
                )
                values = np.asarray(data["values"], dtype=np.float32)
                if values.shape != (len(index), len(fields)):
                    return None
                frame = pd.DataFrame(values, index=index, columns=fields)
                return cls(frame=frame, basket_hash=str(basket_hash))
        except Exception:
            return None

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        index = pd.DatetimeIndex(pd.to_datetime(self.frame.index, utc=True))
        values = self.frame.reindex(columns=list(_MARKET_AGGREGATE_FIELDS)).to_numpy(
            dtype=np.float32,
            copy=False,
        )
        metadata = json.dumps(
            {
                "version": _MARKET_AGGREGATE_STATE_VERSION,
                "basket_hash": self.basket_hash,
                "fields": list(_MARKET_AGGREGATE_FIELDS),
                "last_timestamp": index[-1].isoformat() if len(index) else None,
            },
            sort_keys=True,
        )
        tmp = target.with_suffix(target.suffix + ".tmp")
        np.savez_compressed(
            tmp,
            metadata=np.asarray(metadata),
            index_ns=np.asarray(index.asi8, dtype=np.int64),
            values=values,
        )
        if not tmp.exists() and tmp.with_suffix(tmp.suffix + ".npz").exists():
            tmp = tmp.with_suffix(tmp.suffix + ".npz")
        os.replace(tmp, target)


def _truthy(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def _stable_hash(values: Sequence[str] | None) -> str:
    payload = "\n".join(sorted(str(v) for v in (values or []) if str(v)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def resolve_static_feature_store_id(cfg: Mapping[str, Any], fallback: str) -> str:
    """Return the feature-store identity shared by pipeline and live callers."""

    for key in (
        "static_feature_store_id",
        "feature_store_run_id",
        "live_feature_source_run_id",
        "feature_source_run_id",
        "run_id",
    ):
        value = cfg.get(key)
        if value:
            return str(value)
    return str(fallback)


def configure_static_feature_runtime(
    cfg: Mapping[str, Any] | None,
    *,
    data_root: str | Path | None,
    feature_store_id: str,
    requested_feature_keys: Sequence[str] | None,
    incremental: bool,
    state_scope: str | None = None,
    min_required_ts: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Configure the persisted NumPy rolling-state kernels for one contract.

    Existing explicit state paths are retained.  Otherwise, both training and
    inference use a feature-store-scoped directory so an append-only run can
    reuse its causal rolling and transform state without changing formulas.
    """

    out = dict(cfg or {})
    root = Path(
        str(
            out.get("static_feature_state_root")
            or Path(str(data_root or out.get("data_root") or "data_perp"))
            / "features"
            / str(feature_store_id)
            / "_static_state"
        )
    )
    requested_hash = _stable_hash(requested_feature_keys) if requested_feature_keys else "all"
    scope = state_scope or f"feature_store={feature_store_id}"

    out["static_feature_endpoint_version"] = STATIC_FEATURE_ENDPOINT_VERSION
    out["static_feature_store_id"] = str(feature_store_id)
    out.setdefault("feature_causal_transform_requested_hash", requested_hash)
    out.setdefault("feature_causal_transform_state_scope", scope)
    out.setdefault("feature_raw_rolling_state_scope", scope)

    if incremental:
        causal_enabled = _truthy(
            out.get(
                "feature_causal_transform_state_enabled",
                out.get("live_causal_transform_state_enabled"),
            ),
            default=True,
        )
        rolling_enabled = _truthy(
            out.get(
                "feature_raw_rolling_state_enabled",
                out.get("live_raw_rolling_state_enabled"),
            ),
            default=True,
        )
        out.setdefault("feature_causal_transform_state_enabled", causal_enabled)
        out.setdefault("feature_raw_rolling_state_enabled", rolling_enabled)
        out.setdefault("feature_raw_rolling_state_container_enabled", True)
        # Static feature state is part of the timestamped feature-store
        # contract.  Inference has older, process-local ``live_*`` state
        # paths for non-static caches; letting those win here would make an
        # otherwise identical pipeline/inference append use two independent
        # rolling histories.  Keep the legacy fallback opt-in for old repair
        # tooling, but default the common endpoint to its shared store root.
        allow_legacy_live_paths = _truthy(
            out.get("static_feature_allow_legacy_live_state_path"), default=False
        )
        out.setdefault(
            "feature_causal_transform_state_path",
            str(
                out.get("live_causal_transform_state_path")
                if allow_legacy_live_paths
                and out.get("live_causal_transform_state_path")
                else root / "causal_transform_state.npz"
            ),
        )
        out.setdefault(
            "feature_raw_rolling_state_path",
            str(
                out.get("live_raw_rolling_state_path")
                if allow_legacy_live_paths
                and out.get("live_raw_rolling_state_path")
                else root / "raw_rolling_state.npz"
            ),
        )
        out.setdefault(
            "feature_raw_rolling_state_container_path",
            str(root / "raw_rolling_state.container.sqlite"),
        )
        out.setdefault("market_transform_state_enabled", True)
        out.setdefault(
            "market_transform_state_path",
            str(
                out.get("live_market_transform_state_path")
                if allow_legacy_live_paths
                and out.get("live_market_transform_state_path")
                else root / "market_transform_state.npz"
            ),
        )
        out.setdefault("market_transform_state_scope", scope)
        out.setdefault("feature_raw_rolling_state_sparse_prefix_enabled", True)
        out.setdefault("feature_causal_transform_state_ignore_stale_min_required", True)

    if min_required_ts is not None:
        out.setdefault(
            "feature_causal_transform_min_required_ts",
            pd.Timestamp(min_required_ts).isoformat(),
        )
    return out


def _assert_static_output(features: Mapping[str, Any]) -> None:
    """Prevent model-state outputs from leaking into static materialization."""

    model_state_keys = [
        str(key)
        for key in features
        if str(key).lower().startswith(_MODEL_DERIVED_PREFIXES)
    ]
    if model_state_keys:
        raise RuntimeError(
            "AE/GMM outputs must be applied through the frozen model transform, "
            "not compute_static_features: " + ", ".join(sorted(model_state_keys)[:12])
        )


def compute_static_features(
    panel: Mapping[str, pd.DataFrame],
    mkt_gates: pd.DataFrame,
    cfg: Mapping[str, Any] | None,
    *,
    requested_feature_keys: Sequence[str] | None = None,
    data_root: str | Path | None = None,
    feature_store_id: str | None = None,
    incremental: bool = False,
    state_scope: str | None = None,
    min_required_ts: pd.Timestamp | None = None,
    compute_impl=None,
) -> StaticFeatureResult:
    """Run the sole raw/static feature compute endpoint used by all callers."""

    runtime_cfg = dict(cfg or {})
    store_id = feature_store_id or resolve_static_feature_store_id(
        runtime_cfg,
        fallback="default",
    )
    runtime_cfg = configure_static_feature_runtime(
        runtime_cfg,
        data_root=data_root,
        feature_store_id=store_id,
        requested_feature_keys=requested_feature_keys,
        incremental=incremental,
        state_scope=state_scope,
        min_required_ts=min_required_ts,
    )
    # Local import avoids a module cycle while preserving features.py as the
    # authoritative implementation of every static formula.
    from extreme_price_movements.features import compute_features_hourly

    feature_compute = compute_impl or compute_features_hourly

    features, index, columns = feature_compute(
        dict(panel),
        mkt_gates,
        runtime_cfg,
        requested_feature_keys=(
            [str(key) for key in requested_feature_keys]
            if requested_feature_keys is not None
            else None
        ),
    )
    _assert_static_output(features)
    return StaticFeatureResult(
        features=dict(features),
        index=pd.Index(index),
        columns=[str(column) for column in columns],
        runtime_cfg=runtime_cfg,
    )


def compute_static_market_context(
    panel: Mapping[str, pd.DataFrame],
    basket_syms: Sequence[str],
    *,
    trend_sma_hours: int,
    gate_vol_lookback_hours: int,
    gate_trend_thr: float,
    cfg: Mapping[str, Any] | None = None,
    data_root: str | Path | None = None,
    feature_store_id: str | None = None,
    incremental: bool = False,
) -> StaticMarketContext:
    """Build market transforms through the common causal static endpoint.

    The underlying market transform uses the established NumPy/Numba rolling
    and EWMA kernels.  Incremental calls persist raw cross-sectional aggregates
    and replay the exact market calculation over that compact input history.
    That avoids both repeated wide-panel work and approximate/stale gates.
    """

    from extreme_price_movements.features import add_regime_gates, compute_market_features

    close = panel.get("close")
    required_panels = [panel.get(field) for field in _MARKET_AGGREGATE_FIELDS]
    if not isinstance(close, pd.DataFrame) or any(
        not isinstance(value, pd.DataFrame) for value in required_panels
    ):
        raise ValueError("Market context requires close/high/low/volume panels")
    basket = [str(symbol) for symbol in basket_syms if str(symbol) in close.columns]
    if not basket:
        basket = [str(symbol) for symbol in close.columns]
    basket_hash = _stable_hash(basket)
    aggregate = pd.DataFrame(
        {
            field: panel[field].reindex(index=close.index, columns=basket).mean(axis=1)
            for field in _MARKET_AGGREGATE_FIELDS
        },
        index=close.index,
    ).astype(np.float32)
    aggregate.index = pd.DatetimeIndex(pd.to_datetime(aggregate.index, utc=True))

    runtime_cfg = dict(cfg or {})
    store_id = feature_store_id or resolve_static_feature_store_id(
        runtime_cfg,
        fallback="default",
    )
    runtime_cfg = configure_static_feature_runtime(
        runtime_cfg,
        data_root=data_root,
        feature_store_id=store_id,
        requested_feature_keys=None,
        incremental=incremental,
    )
    state_path = runtime_cfg.get("market_transform_state_path")
    use_state = bool(runtime_cfg.get("market_transform_state_enabled", False)) and bool(
        state_path
    ) and bool(incremental)
    market_input = aggregate
    used_state = False
    if use_state:
        state = MarketAggregateState.load(str(state_path), basket_hash=basket_hash)
        if state is not None and not state.frame.empty:
            market_input = pd.concat([state.frame, aggregate], axis=0, copy=False)
            market_input = market_input.loc[
                ~market_input.index.duplicated(keep="last")
            ].sort_index()
            used_state = True

    synthetic_symbol = "__static_market__"
    market_panel = {
        field: pd.DataFrame(
            {synthetic_symbol: market_input[field].to_numpy(dtype=np.float32)},
            index=market_input.index,
        )
        for field in _MARKET_AGGREGATE_FIELDS
    }
    market_features = compute_market_features(
        market_panel,
        [synthetic_symbol],
        trend_sma_hours=int(trend_sma_hours),
    )
    regime_gates = add_regime_gates(
        market_features,
        gate_vol_lookback_hours=int(gate_vol_lookback_hours),
        gate_trend_thr=float(gate_trend_thr),
    )
    if use_state:
        MarketAggregateState(
            frame=market_input.reindex(columns=list(_MARKET_AGGREGATE_FIELDS)),
            basket_hash=basket_hash,
        ).save(str(state_path))
    market_features = market_features.reindex(aggregate.index)
    regime_gates = regime_gates.reindex(aggregate.index)
    return StaticMarketContext(
        market_features=market_features,
        regime_gates=regime_gates,
        used_state=used_state,
        state_path=str(state_path) if use_state else None,
    )


def _as_feature_array(
    value: Any,
    *,
    index: pd.Index,
    columns: Sequence[str],
) -> np.ndarray | None:
    if isinstance(value, pd.DataFrame):
        return value.reindex(index=index, columns=list(columns)).to_numpy(dtype=np.float32)
    arr = np.asarray(value)
    if arr.ndim == 2 and arr.shape == (len(index), len(columns)):
        return arr.astype(np.float32, copy=False)
    return None


def _atomic_write_block(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
    os.replace(tmp, path)


def materialize_static_feature_blocks(
    features: Mapping[str, Any],
    *,
    index: pd.Index,
    columns: Sequence[str],
    data_root: str | Path,
    feature_store_id: str,
    source: str,
    feature_keys: Sequence[str] | None = None,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
    max_timestamps: int | None = 1,
) -> dict[str, Any]:
    """Persist a timestamp/symbol-keyed Arrow-compatible parity materialization.

    The primary feature store remains the memory-efficient per-symbol Parquet
    base plus DuckDB delta store.  These small, wide blocks are a canonical
    audit surface for training/live comparisons and avoid reconstructing a
    panel before checking one timestamp or one feature family.
    """

    _assert_static_output(features)
    feature_index = pd.DatetimeIndex(pd.to_datetime(index, utc=True, errors="coerce"))
    symbol_columns = [str(column) for column in columns]
    # Callers use zero as an explicit feature-block opt-out.  Treating it as
    # an unlimited timestamp cap would materialize an entire historical panel
    # during an otherwise lightweight incremental update.
    if max_timestamps is not None and int(max_timestamps) <= 0:
        return {"rows": 0, "files": [], "feature_keys": []}
    valid_index = ~feature_index.isna()
    if start_ts is not None:
        valid_index &= feature_index >= pd.Timestamp(start_ts)
    if end_ts is not None:
        valid_index &= feature_index <= pd.Timestamp(end_ts)
    positions = np.flatnonzero(valid_index)
    if max_timestamps is not None and max_timestamps > 0:
        positions = positions[-int(max_timestamps) :]
    if positions.size == 0 or not symbol_columns:
        return {"rows": 0, "files": [], "feature_keys": []}

    requested = [str(key) for key in (feature_keys or features.keys())]
    feature_names = [
        key
        for key in requested
        if key in features
        and _as_feature_array(features[key], index=index, columns=symbol_columns) is not None
    ]
    if not feature_names:
        return {"rows": 0, "files": [], "feature_keys": []}

    root = (
        Path(data_root)
        / "features"
        / str(feature_store_id)
        / "_static_feature_blocks"
    )
    block_id = uuid.uuid4().hex
    written_ns = time.time_ns()
    selected_index = feature_index.take(positions)
    row_count = int(len(selected_index) * len(symbol_columns))
    # Keep reusable column arrays rather than a DataFrame copied once per
    # feature batch.  The static audit surface is normally one timestamp, but
    # multi-bar parity audits can otherwise create a full extra object frame
    # for every group of feature columns.
    base_columns = {
        "ts": np.repeat(selected_index.to_numpy(), len(symbol_columns)),
        "symbol": np.tile(
            np.asarray(symbol_columns, dtype=object), len(selected_index)
        ),
        "__static_written_at_ns": np.full(row_count, written_ns, dtype=np.int64),
        "__static_source__": np.full(row_count, str(source), dtype=object),
    }
    date_values = pd.to_datetime(base_columns["ts"], utc=True).strftime("%Y-%m-%d")
    date_positions = {
        str(date): np.flatnonzero(np.asarray(date_values) == date)
        for date in np.unique(date_values)
    }
    files: list[str] = []
    for offset in range(0, len(feature_names), STATIC_FEATURE_BLOCK_COLUMNS_PER_FILE):
        batch = feature_names[offset : offset + STATIC_FEATURE_BLOCK_COLUMNS_PER_FILE]
        batch_columns = dict(base_columns)
        for key in batch:
            values = _as_feature_array(features[key], index=index, columns=symbol_columns)
            assert values is not None
            batch_columns[key] = values[positions, :].reshape(-1)
        frame = pd.DataFrame(batch_columns, copy=False)
        for date, positions_for_date in date_positions.items():
            date_frame = frame.iloc[positions_for_date]
            path = root / f"date={date}" / f"part-{block_id}-{offset:04d}.parquet"
            _atomic_write_block(date_frame, path)
            files.append(str(path))

    manifest = {
        "version": STATIC_FEATURE_ENDPOINT_VERSION,
        "feature_store_id": str(feature_store_id),
        "source": str(source),
        "created_at_ns": written_ns,
        "rows": row_count,
        "timestamps": int(len(selected_index)),
        "symbols": int(len(symbol_columns)),
        "feature_keys": feature_names,
        "feature_keys_hash": _stable_hash(feature_names),
        "model_state_excluded": ["AE", "GMM"],
        "files": files,
    }
    manifest_dir = root / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{block_id}.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def append_static_features(
    features: Mapping[str, Any],
    *,
    feature_store_ts: pd.Timestamp,
    data_root: str | Path,
    feature_store_id: str | None = None,
    index: pd.Index | None = None,
    columns: Sequence[str] | None = None,
    min_timestamp_by_symbol: Mapping[str, pd.Timestamp] | None = None,
    save_workers: int | None = None,
    replace_existing: bool = False,
    overwrite_columns: set[str] | Sequence[str] | None = None,
    source: str,
    block_max_timestamps: int | None = 1,
) -> dict[str, Any]:
    """Append static features through the shared Parquet/DuckDB store contract.

    ``save_features`` selects DuckDB for incremental updates and compacts to the
    per-symbol Parquet base only after the configured delta budget.  Keeping
    that implementation behind this endpoint makes training and inference
    writes share the same idempotence, schema, and numeric downcast behavior.
    """

    if not features:
        return {"saved": False, "block": {"rows": 0, "files": []}}
    from extreme_price_movements.data_store import save_features

    ts = pd.Timestamp(feature_store_ts)
    store_id = str(feature_store_id or ts.strftime("%Y%m%d_%H%M%S"))
    effective_save_workers = (
        resolve_static_feature_save_workers()
        if save_workers is None
        else max(2, min(int(save_workers), 32))
    )
    save_features(
        dict(features),
        ts,
        str(data_root),
        min_timestamp_by_symbol=(
            dict(min_timestamp_by_symbol) if min_timestamp_by_symbol else None
        ),
        feat_index=index,
        feat_columns=list(columns) if columns is not None else None,
        save_workers=effective_save_workers,
        replace_existing=bool(replace_existing),
        overwrite_columns=set(overwrite_columns or set()),
    )

    if index is None or columns is None:
        first = next(iter(features.values()))
        if isinstance(first, pd.DataFrame):
            index = first.index
            columns = list(first.columns)
        else:
            return {"saved": True, "block": {"rows": 0, "files": []}}
    block = materialize_static_feature_blocks(
        features,
        index=index,
        columns=columns,
        data_root=data_root,
        feature_store_id=store_id,
        source=source,
        feature_keys=sorted(str(key) for key in features),
        max_timestamps=block_max_timestamps,
    )
    return {"saved": True, "block": block}


def read_static_features(
    *,
    feature_store_ts: pd.Timestamp,
    data_root: str | Path,
    feature_keys: Sequence[str] | None = None,
    symbols: Sequence[str] | None = None,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
    allowed_periods=None,
    output_layout: str = "panels",
) -> dict | pd.DataFrame | None:
    """Read static features through the same Parquet-plus-DuckDB view.

    This is intentionally a thin endpoint over the selective store reader;
    that reader already merges the Parquet base, Parquet repair parts, and the
    DuckDB append buffer before pivoting to feature panels.
    """

    from extreme_price_movements.data_store import load_features_selected

    loaded = load_features_selected(
        pd.Timestamp(feature_store_ts),
        str(data_root),
        feature_keys=list(feature_keys) if feature_keys is not None else None,
        symbols=list(symbols) if symbols is not None else None,
        start_ts=start_ts,
        end_ts=end_ts,
        allowed_periods=allowed_periods,
    )
    layout = str(output_layout or "panels").strip().lower()
    if layout == "panels" or loaded is None:
        return loaded
    if layout != "symbol_frame":
        raise ValueError(
            f"Unsupported static feature output layout {output_layout!r}; "
            "expected 'panels' or 'symbol_frame'"
        )
    requested_symbols = [str(symbol) for symbol in (symbols or [])]
    if len(requested_symbols) != 1:
        raise ValueError("symbol_frame output requires exactly one requested symbol")
    if not hasattr(loaded, "symbol_frame"):
        raise TypeError("Shared static reader does not support symbol-frame materialization")
    return loaded.symbol_frame(requested_symbols[0], keys=feature_keys)


def read_static_feature_blocks(
    *,
    data_root: str | Path,
    feature_store_id: str,
    feature_keys: Sequence[str],
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
    source: str | None = None,
    include_metadata: bool = False,
) -> pd.DataFrame:
    """Read a timestamp/symbol keyed parity block, coalescing append batches.

    ``source`` retains the independently materialized training/pipeline or
    inference surface for numeric parity comparisons.  The default preserves
    the latest-write view used by lightweight feature inspection.
    """

    root = Path(data_root) / "features" / str(feature_store_id) / "_static_feature_blocks"
    if not root.exists():
        return pd.DataFrame(columns=["ts", "symbol", *feature_keys])
    requested = [str(key) for key in feature_keys]
    base_columns = [
        "ts",
        "symbol",
        "__static_written_at_ns",
        "__static_source__",
        *requested,
    ]
    def _utc_partition_day(value: pd.Timestamp | None) -> str | None:
        if value is None:
            return None
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.strftime("%Y-%m-%d")

    # Static blocks are partitioned by UTC decision date.  Prune at the
    # directory level before inspecting Parquet schemas; parity checks often
    # target one or a few bars while the store can contain many historical
    # writes from both training and inference.
    start_day = _utc_partition_day(start_ts)
    end_day = _utc_partition_day(end_ts)
    part_paths: list[Path] = []
    for date_dir in sorted(root.glob("date=*")):
        date_value = date_dir.name.removeprefix("date=")
        if start_day is not None and date_value < start_day:
            continue
        if end_day is not None and date_value > end_day:
            continue
        part_paths.extend(sorted(date_dir.glob("*.parquet")))

    frames: list[pd.DataFrame] = []
    for path in part_paths:
        try:
            import pyarrow.parquet as pq

            available = set(pq.ParquetFile(path).schema.names)
            cols = [column for column in base_columns if column in available]
            if not {"ts", "symbol"}.issubset(cols):
                continue
            frames.append(pd.read_parquet(path, columns=cols))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["ts", "symbol", *requested])
    out = pd.concat(frames, axis=0, ignore_index=True, copy=False)
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["ts", "symbol"])
    if start_ts is not None:
        out = out.loc[out["ts"] >= pd.Timestamp(start_ts)]
    if end_ts is not None:
        out = out.loc[out["ts"] <= pd.Timestamp(end_ts)]
    if source is not None:
        if "__static_source__" not in out.columns:
            return pd.DataFrame(columns=["ts", "symbol", *requested])
        out = out.loc[out["__static_source__"] == str(source)]
    out = out.sort_values("__static_written_at_ns", kind="stable")
    grouped = out.groupby(["ts", "symbol"], as_index=False, sort=True)
    out = grouped.last()
    for key in requested:
        if key not in out.columns:
            out[key] = np.nan
    result_columns = ["ts", "symbol", *requested]
    if include_metadata:
        for key in ("__static_source__", "__static_written_at_ns"):
            if key in out.columns:
                result_columns.append(key)
    return out[result_columns].sort_values(["ts", "symbol"])


def compare_static_feature_block_sources(
    *,
    data_root: str | Path,
    feature_store_id: str,
    feature_keys: Sequence[str],
    left_source: str,
    right_source: str,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Compare independently materialized static features on identical rows.

    This deliberately compares only timestamp/symbol rows that both sources
    emitted.  Feature availability can legitimately differ while a new store
    is warming up; the metric must identify that separately instead of treating
    it as an arithmetic mismatch.
    """

    requested = [str(key) for key in feature_keys]
    left = read_static_feature_blocks(
        data_root=data_root,
        feature_store_id=feature_store_id,
        feature_keys=requested,
        start_ts=start_ts,
        end_ts=end_ts,
        source=left_source,
    )
    right = read_static_feature_blocks(
        data_root=data_root,
        feature_store_id=feature_store_id,
        feature_keys=requested,
        start_ts=start_ts,
        end_ts=end_ts,
        source=right_source,
    )
    keys = ["ts", "symbol"]
    overlap = left.merge(right, on=keys, how="inner", suffixes=("__left", "__right"))
    per_feature: dict[str, dict[str, Any]] = {}
    all_within_tolerance = True
    for key in requested:
        left_col = f"{key}__left"
        right_col = f"{key}__right"
        if left_col not in overlap.columns or right_col not in overlap.columns:
            per_feature[key] = {
                "finite_pairs": 0,
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "within_tolerance": False,
            }
            all_within_tolerance = False
            continue
        left_values = pd.to_numeric(overlap[left_col], errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        )
        right_values = pd.to_numeric(overlap[right_col], errors="coerce").to_numpy(
            dtype=np.float64, copy=False
        )
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        if not bool(finite.any()):
            per_feature[key] = {
                "finite_pairs": 0,
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "within_tolerance": True,
            }
            continue
        delta = np.abs(left_values[finite] - right_values[finite])
        tolerance = atol + rtol * np.abs(left_values[finite])
        within = bool(np.all(delta <= tolerance))
        per_feature[key] = {
            "finite_pairs": int(finite.sum()),
            "max_abs_diff": float(delta.max()),
            "mean_abs_diff": float(delta.mean()),
            "within_tolerance": within,
        }
        all_within_tolerance = all_within_tolerance and within
    return {
        "feature_store_id": str(feature_store_id),
        "left_source": str(left_source),
        "right_source": str(right_source),
        "left_rows": int(len(left)),
        "right_rows": int(len(right)),
        "overlap_rows": int(len(overlap)),
        "all_within_tolerance": bool(all_within_tolerance),
        "per_feature": per_feature,
    }
