#!/usr/bin/env python3
"""Append only the newest strict-R3 feature rows from a cached source panel.

This challenger uses the repository's production live feature-cache engine,
not the retired simplified feature path.  Existing candidate-keyed feature
rows are immutable; only signal timestamps newer than the prefix are emitted
by the compute engine and appended.  Promotion requires exact comparison with
the canonical full materializer.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import (  # noqa: E402
    CFG,
    SESSION_CALENDAR_BASE_CANDIDATE_FEATURE_KEYS,
)
from extreme_price_movements.features import (  # noqa: E402
    _apply_derived_feature_history_state,
    _add_regime_panel_composite_features,
    _regime_composite_group_from_key,
    _regime_composite_parent_from_key,
    _safe_log_df,
    add_regime_gates,
    compute_features_hourly,
    compute_market_features,
)
from extreme_price_movements import fast_funcs as ff  # noqa: E402
from extreme_price_movements.inference.strict_r3_final14_state import (  # noqa: E402
    FINAL14_FIELD_ORDER,
)
from extreme_price_movements.inference.orderbook_feature_state import (  # noqa: E402
    ORDERBOOK_OUTPUTS,
)
from scripts.materialize_strict_r3_forward_features import (  # noqa: E402
    _IDENTITY_COLUMNS,
    _repair_cross_asset_state_fields,
    _refresh_feature_coverage,
)
from scripts.run_tp6_sl4_exact170_canonical_consensus import (  # noqa: E402
    FROZEN_GENERATION_DEPENDENCIES,
    _load_contract,
)
from scripts.update_strict_r3_feature_panel_state import STATE_SCHEMA  # noqa: E402
from scripts.restore_strict_r3_feature_state_bundle import (  # noqa: E402
    restore_bundle,
)


# The P8U source builder persists the same contemporaneous primitive-panel
# shape as the older strict-R3 panel-state writer, but intentionally uses a
# distinct schema to make its source identity auditable.  Accepting it here is
# safe only after the structural checks below; a bare schema alias must never
# bypass the complete-universe and contiguous-hourly requirements.
P8U_CANONICAL_SOURCE_STATE_SCHEMA = "strict_r3_p8u_canonical_source_panel_state_v1"

# ``rv_24h`` is a raw rolling primitive: its causal history is owned by the
# raw-rolling operator state, not by the recursive derived-feature overlay.
# Including it in the latter only on append requests changed the SQLite state
# identity even though the canonical bootstrap intentionally never stored it.
# Keep the exclusion explicit so bootstrap and one-hour append runs reopen the
# same nested-derived namespace.
NESTED_DERIVED_RAW_PRIMITIVE_KEYS = frozenset({"rv_24h"})


def _compute_exact_volatility_zscore(
    panel: Mapping[str, pd.DataFrame],
    *,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Return the canonical terminal ``volatility_zscore`` from its source.

    This is deliberately a *single-parent* repair for bounded inference.  The
    selected field is the causal robust z-score of log ATR in log-price space.
    Its 720-hour rolling distribution is stateful, and a few ULPs in a resumed
    ATR path can be magnified around a small MAD.  Rebuilding only this parent
    from the full causal source is cheap (three EWMAs plus three rolling
    primitives) and exactly preserves the batch feature contract.  It is not
    a full feature-graph fallback and never consumes data after the requested
    decision timestamp.

    Keep this algebra byte-for-byte aligned with the canonical constructors in
    ``compute_features_hourly``: log high/low use the span-5 native EWMA,
    ATR uses its native Wilder alpha, and the final robust-z/support/dispersion
    implementation uses the same compiled kernels and constants.
    """
    required = ("close", "high", "low")
    missing = [name for name in required if not isinstance(panel.get(name), pd.DataFrame)]
    if missing:
        raise KeyError(f"volatility_zscore source panel lacks {missing}")

    safe_log_eps = float(cfg.get("safe_log_eps", 1e-12))
    high_log = _safe_log_df(panel["high"].astype(np.float32), eps=safe_log_eps)
    low_log = _safe_log_df(panel["low"].astype(np.float32), eps=safe_log_eps)
    close_log = _safe_log_df(panel["close"].astype(np.float32), eps=safe_log_eps)
    high_smooth = ff.numba_ewma(high_log, 2.0 / 6.0, False)
    low_smooth = ff.numba_ewma(low_log, 2.0 / 6.0, False)
    prev_close = close_log.shift(1)
    true_range = np.maximum(
        high_smooth - low_smooth,
        np.maximum((high_smooth - prev_close).abs(), (low_smooth - prev_close).abs()),
    )
    atr_log = ff.numba_ewma(
        true_range, 1.0 / float(cfg.get("atr_n", 14)), False
    ).clip(lower=float(cfg.get("atr_ln_floor", 1e-6)))
    source = np.log(atr_log.where(atr_log > 0.0)).astype(np.float32)
    window = int(cfg.get("feature_portability_repair_window_hours", 24 * 30))
    raw = pd.DataFrame(
        ff.numba_rolling_robust_zscore(
            source.to_numpy(dtype=np.float32, copy=False), window
        ),
        index=source.index,
        columns=source.columns,
    )
    support = ff.numba_rolling_sum(source.notna().astype(np.float32), window)
    dispersion = ff.apply_to_frame(
        source, ff._numba_rolling_std_nan_safe, window
    )
    minimum = max(10, min(window, window // 4))
    valid = (
        source.notna()
        & (support >= float(minimum))
        & np.isfinite(dispersion)
        & (dispersion > np.float32(1e-8))
    )
    return raw.where(valid).clip(-8.0, 8.0).astype(np.float32)


def _validate_panel_state_for_materialization(state: object) -> str:
    """Validate a source-only panel state and return its sealed schema.

    Both accepted formats contain primitive, target-free source frames.  The
    canonical P8U state is not rewritten or converted: its complete panel is
    consumed verbatim after validation, so its hash and source identity remain
    the upstream contract.
    """
    if not isinstance(state, Mapping):
        raise ValueError("unsupported source-panel state")
    schema = str(state.get("schema") or "")
    if schema == STATE_SCHEMA:
        return schema
    if schema != P8U_CANONICAL_SOURCE_STATE_SCHEMA:
        raise ValueError("unsupported source-panel state")
    panel = state.get("panel")
    symbols = tuple(map(str, state.get("symbols") or ()))
    if not isinstance(panel, Mapping) or not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("canonical P8U source state lacks a unique panel universe")
    close = panel.get("close")
    if not isinstance(close, pd.DataFrame) or close.empty:
        raise ValueError("canonical P8U source state lacks a non-empty close panel")
    index = pd.DatetimeIndex(close.index)
    expected = pd.date_range(index[0], index[-1], freq="h", tz="UTC")
    if not index.equals(expected):
        raise ValueError("canonical P8U source state close panel is not contiguous hourly")
    if list(map(str, close.columns)) != list(symbols):
        raise ValueError("canonical P8U source state close universe mismatch")
    for name, frame in panel.items():
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"canonical P8U source field {name} is not a frame")
        if not pd.DatetimeIndex(frame.index).equals(index):
            raise ValueError(f"canonical P8U source field {name} index mismatch")
        if list(map(str, frame.columns)) != list(symbols):
            raise ValueError(f"canonical P8U source field {name} universe mismatch")
    if not state.get("canonical_manifest_sha256"):
        raise ValueError("canonical P8U source state lacks a manifest identity")
    return schema


def _frozen_causal_transform_feature_order(bundle_dir: Path) -> list[str]:
    """Return the widest frozen transform workset from an immutable bundle.

    Older transition bundles can contain both the active namespace and a
    narrower namespace created while one raw input was unavailable.  The
    canonical transform contract is the widest hash-bound workset.  Missing
    members are advanced as NaN, so ordinary source absence cannot change the
    namespace or reset rolling history.
    """
    candidates: list[tuple[int, str, list[str]]] = []
    for path in sorted((Path(bundle_dir) / "states").glob(
        "causal_transform_state.*.npz"
    )):
        with np.load(path, allow_pickle=True) as state:
            metadata = json.loads(str(state["metadata"].item()))
        order = [str(value) for value in metadata.get("feature_order", [])]
        if order:
            candidates.append((len(order), path.name, order))
    if not candidates:
        raise ValueError(
            "persisted-state bundle lacks a causal-transform feature order"
        )
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]


def _prune_transitional_causal_transform_copies(
    cache_dir: Path,
    *,
    expected_order: list[str],
) -> list[str]:
    """Remove narrower transition namespaces from the private transaction.

    The immutable source bundle is never modified.  Retaining a narrower
    namespace would make the next snapshot publish two different semantic
    worksets and eventually fail on their diverging watermarks.
    """
    removed: list[str] = []
    for path in sorted(Path(cache_dir).glob("causal_transform_state.*.npz")):
        with np.load(path, allow_pickle=True) as state:
            metadata = json.loads(str(state["metadata"].item()))
        order = [str(value) for value in metadata.get("feature_order", [])]
        if order != expected_order:
            path.unlink()
            removed.append(path.name)
    return removed


# These fields are the exact residual from the isolated 2026-08-14 append
# audit: their canonical definitions contain expanding/long-memory operators
# not yet represented by RawRollingFeatureState.  The hybrid producer computes
# only this closure on the sealed full panel and all other fields on the
# append-state tail.  This is a correctness allow-list, not an approximation;
# every produced row is still compared with the canonical producer before the
# mode can be bound into an inference bundle.
EXACT_LONG_MEMORY_FIELDS = frozenset({
    # The canonical long contract uses these OI/liquidation state fields.
    # Their rolling horizons exceed the 72-hour append tail, so leaving them
    # on that tail silently turns available causal source history into NaNs.
    # They therefore share the read-only complete-panel closure below; this
    # is the same operator family used in batch training, not an imputation.
    "asset_minus_mkt_long_flush_intensity_4h",
    "bars_to_resistance_daily_vwap",
    "bars_to_resistance_daily_donchian",
    "down_barrier_pressure_daily_donchian",
    "eig_effective_rank__open_interest",
    "grind_score_surprise",
    "excess_6h_ts_resid",
    "log_bars_since_below_1atr",
    "log_bars_since_below_3atr",
    "memory_asymmetry_1ATR",
    "memory_asymmetry_3ATR",
    "mkt_oi_dispersion_24h",
    "mkt_oi_flush_z_30d",
    "mkt_pct_oi_chg_1h_rz_lt_minus1",
    "mkt_pct_oi_chg_4h_rz_lt_minus2",
    "mkt_pct_price_down_oi_down_1h",
    "mkt_pct_price_up_oi_down_1h",
    "mkt_pct_price_up_oi_down_4h",
    "mkt_pct_price_up_oi_up_1h",
    "mkt_pct_price_up_oi_up_4h",
    "negative_breadth_pct",
    "ob_depth_l20_to_qv_z_7d",
    "ob_spread_bps",
    "ob_spread_bps_z_24h",
    "ob_spread_z_24h",
    "ob_trade_size_to_l1_depth_z_24h",
    "prior_volatility",
    "price_recovery_oi_still_falling_1h",
    "price_rv_15d_robust_z",
    "post_flush_leverage_rebuild",
    "post_liquidation_rebound_score",
    "pct_assets_extreme_oi_drop_1h",
    "liquidation_climax_score",
    "oi_drop_acceleration_4h_rz",
    "q_iqr__ob_trade_size_to_l1_depth_z_24h",
    "q_iqr__amihud_z_peer_resid",
    "q_iqr__ret48h_bench_resid",
    "q_lower_tail__ob_depth_usd_l20_z",
    "q_lower_tail__volume_z_24",
    "q_lower_tail__ob_spread_bps_z_24h",
    "q_tail_asym__ob_depth_usd_l20_z",
    "q_tail_asym__ob_spread_bps_z_24h",
    "q_tail_asym__amihud_z_peer_resid",
    "q_tail_asym__vol_z_4h",
    "q_tail_width__volume_z_12",
    "q_tail_width__oi_to_volume_7d_z_180d",
    "q_upper_tail__bars_in_high_vol_state_log_norm",
    "q_upper_tail__ob_spread_bps_z_24h",
    "ret4h_peer_resid",
    "spike_score_surprise",
    "state_spectral_eig_condition",
    "state_spectral_eig_gap_1_2",
    "state_spectral_eig_top3_share",
    "volume_percentile",
    "volume_price_corr_ts_resid",
    "volume_z_12",
    "volume_z_24",
    "xasset_mkt_spread_bps",
    "xs_dispersion__efficiency_ratio_20",
    "xs_dispersion__ob_depth_to_qv_z_x_rvol_z",
    "xs_dispersion__ob_spread_bps_z_24h",
    "xs_dispersion__oi_to_volume_7d_z_180d",
    "xs_dispersion__rvol_z",
    "xs_dispersion__vol_z",
    "xs_dispersion__volume_zscore_48h",
    "xs_dispersion__xasset_ob_liquidity_divergence_z_24h",
    "xs_dispersion__xasset_ob_liquidity_ts_resid",
})

PRICE_MEMORY_STATE_FIELD_ORDER = (
    "log_bars_since_above_1atr",
    "log_bars_since_below_1atr",
    "memory_asymmetry_1ATR",
    "log_bars_since_above_2atr",
    "log_bars_since_below_2atr",
    "memory_asymmetry_2ATR",
    "log_bars_since_above_3atr",
    "log_bars_since_below_3atr",
    "memory_asymmetry_3ATR",
    "prior_volatility",
    "volume_percentile",
    "bars_to_resistance_daily_vwap",
    "bars_to_resistance_daily_donchian",
    "down_barrier_pressure_daily_donchian",
)
PRICE_MEMORY_STATE_FIELDS = frozenset(PRICE_MEMORY_STATE_FIELD_ORDER)
# These four fields remain on the exact fallback until their real-panel output
# contract is reconciled. The raw state is exact, but the deployed historical
# representation differs at the final transform/missingness boundary.
PRICE_MEMORY_PROMOTED_FIELDS = PRICE_MEMORY_STATE_FIELDS.difference({
    "bars_to_resistance_daily_vwap",
    "bars_to_resistance_daily_donchian",
    "down_barrier_pressure_daily_donchian",
    "memory_asymmetry_3ATR",
})
RESIDUAL_SURPRISE_STATE_FIELDS = frozenset({
    "excess_6h_ts_resid",
    "spike_score_surprise",
    "grind_score_surprise",
    "volume_price_corr_ts_resid",
})
CROSS_SECTIONAL_STATE_FIELDS = frozenset(
    name
    for name in EXACT_LONG_MEMORY_FIELDS
    if name.startswith((
        "xs_dispersion__", "q_lower_tail__", "q_upper_tail__",
        "q_iqr__", "q_tail_width__", "q_tail_asym__",
    ))
)


def _latest_matrix(
    features: dict[str, pd.DataFrame],
    *,
    candidates: pd.DataFrame,
    requested: list[str],
) -> pd.DataFrame:
    identities = candidates.loc[:, _IDENTITY_COLUMNS].copy()
    identities["__ts__"] = pd.to_datetime(identities["__ts__"], utc=True)
    value_columns: dict[str, np.ndarray] = {}
    timestamps = pd.to_datetime(identities["__ts__"], utc=True)
    symbols = identities["__symbol__"].astype(str)
    # The old implementation reindexed one feature row per timestamp.  On a
    # long historical bootstrap that is fields × hours Python-level reindexes
    # even though the requested matrix is a simple point lookup.  Build the
    # common timestamp/symbol index once, then gather each field vectorially.
    # ``factorize(sort=False)`` preserves the candidate order, so this is
    # numerically and causally identical to the previous per-timestamp loop.
    timestamp_codes, unique_timestamps = pd.factorize(timestamps, sort=False)
    symbol_codes, unique_symbols = pd.factorize(symbols, sort=False)
    for field in requested:
        frame = features.get(field)
        if not isinstance(frame, pd.DataFrame):
            value_columns[field] = np.full(len(identities), np.nan, dtype=np.float32)
            continue
        aligned = frame.reindex(index=unique_timestamps, columns=unique_symbols)
        matrix = aligned.to_numpy(dtype=np.float32, copy=False)
        value_columns[field] = matrix[timestamp_codes, symbol_codes]
    return pd.concat(
        [identities.reset_index(drop=True), pd.DataFrame(value_columns)], axis=1
    )


def _apply_current_source_missingness(
    latest: pd.DataFrame,
    *,
    complete_panel: dict[str, pd.DataFrame],
    cfg: dict,
) -> None:
    """Apply the frozen current-source availability contract in-place.

    This was previously inline in ``main``.  Keeping it as a helper permits
    chronological, bounded output batches without changing any value-level
    source availability semantics.
    """
    bid = complete_panel.get("orderbook_best_bid")
    ask = complete_panel.get("orderbook_best_ask")
    if not isinstance(bid, pd.DataFrame) or not isinstance(ask, pd.DataFrame):
        return
    shift_bars = max(0, int(cfg.get("microstructure_shift_bars", 1)))
    for current_ts, positions in latest.groupby("__ts__", sort=False).groups.items():
        current_ts = pd.Timestamp(current_ts)
        source_ts = current_ts - pd.Timedelta(hours=shift_bars)
        if source_ts not in bid.index or source_ts not in ask.index:
            continue
        positions = np.asarray(list(positions), dtype=np.int64)
        symbols_now = latest.iloc[positions]["__symbol__"].astype(str)
        # Match the frozen feature graph for ordinary book fields: it
        # causally carries the last observed book and then applies the
        # declared one-bar shift.
        current_bid = bid.loc[:source_ts].ffill().iloc[-1].reindex(
            symbols_now.to_list()
        )
        current_ask = ask.loc[:source_ts].ffill().iloc[-1].reindex(
            symbols_now.to_list()
        )
        bid_values = current_bid.to_numpy(dtype=float)
        ask_values = current_ask.to_numpy(dtype=float)
        spread_unavailable = ~(
            np.isfinite(bid_values)
            & np.isfinite(ask_values)
            & (bid_values > 0.0)
            & (ask_values > bid_values)
        )
        for name in ("ob_spread_bps", "ob_spread_bps_z_24h", "ob_spread_z_24h"):
            if name in latest.columns:
                latest.loc[positions[spread_unavailable], name] = np.nan

        def source_values(name: str) -> np.ndarray:
            frame = complete_panel.get(name)
            if not isinstance(frame, pd.DataFrame) or source_ts not in frame.index:
                return np.full(len(symbols_now), np.nan, dtype=float)
            return frame.loc[source_ts].reindex(symbols_now.to_list()).to_numpy(
                dtype=float
            )

        exact_bid = source_values("orderbook_best_bid")
        exact_ask = source_values("orderbook_best_ask")
        bid_q1 = source_values("orderbook_bid_qty_1")
        ask_q1 = source_values("orderbook_ask_qty_1")
        # Only the authorised coarse trade-size field has the stricter
        # same-source-hour availability contract. Its rolling mean-trade
        # parent follows the frozen graph's causal carry, but the current
        # L1 book itself must exist and have positive depth; otherwise the
        # row receives the same-timestamp complete-universe median.
        trade_available = (
            np.isfinite(exact_bid)
            & np.isfinite(exact_ask)
            & (exact_bid > 0.0)
            & (exact_ask > exact_bid)
            & np.isfinite(bid_q1)
            & np.isfinite(ask_q1)
            & ((bid_q1 + ask_q1) > 0.0)
        )
        if "ob_depth_l20_to_qv_z_7d" in latest.columns:
            latest.loc[
                positions[spread_unavailable], "ob_depth_l20_to_qv_z_7d"
            ] = np.nan
        if "ob_trade_size_to_l1_depth_z_24h" in latest.columns:
            latest.loc[
                positions[~trade_available], "ob_trade_size_to_l1_depth_z_24h"
            ] = np.nan


def _copy_state_file(source: str, target: str) -> str:
    """Copy state privately so even an in-place writer cannot touch canonical."""
    return shutil.copy2(source, target)


def _begin_state_transaction(cache_dir: Path) -> tuple[Path, Path, bool]:
    """Create a private state workspace for every state-mutating run.

    The returned workspace is deliberately a sibling of the canonical cache
    so publication can use an atomic rename.  It is *not* created when the
    canonical cache is absent: restore_bundle requires a nonexistent target,
    while an ordinary bootstrap will create it through the feature writers.
    """
    working = cache_dir.with_name(cache_dir.name + f".txn.{os.getpid()}")
    if working.exists():
        raise FileExistsError(f"stale state transaction exists: {working}")
    existed = cache_dir.exists()
    if existed:
        shutil.copytree(cache_dir, working, copy_function=_copy_state_file)
    return working, cache_dir, existed


def _commit_state_transaction(
    working: Path, canonical: Path, canonical_existed: bool,
) -> None:
    if not working.exists():
        raise FileNotFoundError(f"state transaction produced no cache: {working}")
    if not canonical_existed:
        if canonical.exists():
            raise FileExistsError(
                f"canonical cache appeared during transaction: {canonical}"
            )
        working.rename(canonical)
        return
    backup = canonical.with_name(canonical.name + f".previous.{os.getpid()}")
    if backup.exists():
        raise FileExistsError(f"stale state transaction backup exists: {backup}")
    canonical.rename(backup)
    try:
        working.rename(canonical)
    except Exception:
        backup.rename(canonical)
        raise
    shutil.rmtree(backup)


def _compute_contract_features(
    panel: dict[str, pd.DataFrame],
    *,
    symbols: list[str],
    requested: list[str],
    cfg: dict,
    market_gate_panel: dict[str, pd.DataFrame] | None = None,
    volatility_zscore_panel: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.Index, pd.Index]:
    """Run the canonical constructors for one declared feature workset.

    The market-volatility selector is a tiny, shared stateful parent for the
    adaptive ATR choice.  Reconstructing it from a bounded source tail changes
    the arbitrary level of its cumulative market index and, through floating
    point cancellation in rolling volatility, can move the selected ATR by a
    few ULPs.  That in turn amplifies in ATR-normalised model fields.

    ``market_gate_panel`` is therefore allowed to supply the complete causal
    source only for this inexpensive market-gate calculation.  The same holds
    for the one numerically sensitive log-ATR robust-z parent supplied through
    ``volatility_zscore_panel``.  The actual 185-field feature graph still
    runs on ``panel`` and remains bounded.  These are exact targeted parent
    repairs, not a full-graph fallback.
    """
    market_source = market_gate_panel if market_gate_panel is not None else panel
    market = compute_market_features(market_source, symbols)
    gates = add_regime_gates(
        market, gate_vol_lookback_hours=24 * 7, gate_trend_thr=0.0,
    )
    if market_gate_panel is not None:
        feature_index = panel["close"].index
        market = market.reindex(feature_index)
        gates = gates.reindex(feature_index)
    # The canonical materializer performs one final composite repair after the
    # generic feature transform. Persist derived-parent history only at that
    # final semantic boundary; using the same state inside both calls would mix
    # pre-transform and post-transform parent representations.
    inner_cfg = dict(cfg)
    inner_cfg["feature_market_spectral_history_state_enabled"] = False
    inner_cfg["live_market_spectral_history_state_enabled"] = False
    inner_cfg["feature_derived_regime_history_state_enabled"] = False
    features, feature_index, feature_columns = compute_features_hourly(
        panel, gates, inner_cfg, requested_feature_keys=requested,
    )
    if volatility_zscore_panel is not None and "volatility_zscore" in requested:
        exact_volatility = _compute_exact_volatility_zscore(
            volatility_zscore_panel, cfg=cfg
        ).reindex(index=feature_index, columns=feature_columns)
        features["volatility_zscore"] = exact_volatility.astype(np.float32)
    if (
        not isinstance(features.get("ob_spread_bps_z_24h"), pd.DataFrame)
        and isinstance(features.get("ob_spread_z_24h"), pd.DataFrame)
    ):
        features["ob_spread_bps_z_24h"] = features["ob_spread_z_24h"].astype(
            np.float32
        )
    spread = features.get("ob_spread_bps")
    if isinstance(spread, pd.DataFrame) and not spread.empty:
        basket_bases = {
            str(value).split("/", 1)[0].upper()
            for value in cfg.get("market_basket", [])
        }
        basket = [
            symbol for symbol in spread.columns
            if str(symbol).split("/", 1)[0].upper() in basket_bases
        ]
        source = spread[basket] if basket else spread
        market_spread = source.mean(axis=1, skipna=True).astype(np.float32)
        features["xasset_mkt_spread_bps"] = pd.DataFrame(
            np.broadcast_to(
                market_spread.to_numpy(dtype=np.float32)[:, None],
                (len(market_spread), len(spread.columns)),
            ),
            index=spread.index,
            columns=spread.columns,
        ).astype(np.float32, copy=False)
    regime_parent_keys: set[str] = set()
    group_map = (cfg or {}).get("MODEL_REGIME_COMPOSITE_EIGEN_GROUPS", {}) or {}
    for name in requested:
        parent = _regime_composite_parent_from_key(str(name))
        if parent:
            regime_parent_keys.add(parent)
        group = _regime_composite_group_from_key(str(name))
        if group:
            regime_parent_keys.update(str(value) for value in group_map.get(group, []))
    regime_parent_keys.update(
        str(value)
        for value in (cfg or {}).get("MARKET_SPECTRAL_POSITION_SOURCE_FEATURE_KEYS", [])
    )
    _apply_derived_feature_history_state(
        features,
        regime_parent_keys,
        cfg,
        stage="pre_regime",
        index=pd.Index(feature_index),
        columns=pd.Index(feature_columns),
    )
    # ``compute_features_hourly`` already creates canonical composites from
    # their raw parents before the generic transform.  The outer dependency
    # pass may fill genuinely absent requested composites, but must never
    # overwrite an existing one using post-transform parents.
    missing_requested = {name for name in requested if name not in features}
    _add_regime_panel_composite_features(
        features,
        missing_requested,
        cfg,
        pd.Index(feature_index),
        pd.Index(feature_columns),
    )
    return features, pd.Index(feature_index), pd.Index(feature_columns)


def _frozen_spectral_parent_keys(path: Path) -> list[str]:
    """Return the exact primitive parents required by a frozen spectral state.

    Spectral source columns are named ``<feature>__<summary>``.  The feature
    constructors are request-driven, so loading the frozen column contract is
    insufficient unless its primitive parents are also included in the
    internal compute workset.  These parents are dependencies only; they do
    not expand the persisted model feature contract.
    """
    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    if payload.get("schema") != "strict_r3_market_spectral_source_state_v1":
        raise ValueError("unsupported market-spectral source state")
    return list(dict.fromkeys(
        str(column).rsplit("__", 1)[0]
        for column in payload.get("selected_columns", ())
        if "__" in str(column)
    ))


def main() -> None:
    run_started = time.perf_counter()
    phase_timings: dict[str, float] = {}

    def _mark_phase(name: str, started: float) -> float:
        now = time.perf_counter()
        phase_timings[name] = float(now - started)
        return now

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--panel-state", type=Path, required=True)
    parser.add_argument("--feature-prefix", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-is-already-private",
        action="store_true",
        help=(
            "Offline orchestration only: cache-dir is already a timestamped, "
            "unpublished private copy. Skip the materializer's inner cache "
            "clone; the caller must publish its state pointer only after all "
            "post-materialisation checks pass. Never use for a live cache."
        ),
    )
    parser.add_argument(
        "--requested-features-json",
        type=Path,
        help=(
            "A sealed feature-plan JSON. A top-level full_union array is "
            "requested exactly; this avoids widening P8U inference to the "
            "legacy all-feature contract."
        ),
    )
    parser.add_argument(
        "--feature-cache-namespace",
        help=(
            "Explicit isolated state namespace for a sealed model contract. "
            "Never reuse a cache namespace across different feature unions."
        ),
    )
    parser.add_argument(
        "--restore-state-bundle",
        type=Path,
        help=(
            "Restore this immutable causal-state snapshot when --cache-dir "
            "does not yet exist. This is shared by chronological training "
            "chunks and hourly inference."
        ),
    )
    parser.add_argument("--expected-state-contract-hash")
    parser.add_argument(
        "--stateful-tail-hours", type=int,
        help=(
            "Use only this causal primitive tail after an exact full-history "
            "state bootstrap. Raw rolling and causal-transform states must "
            "already cover the immediately preceding timestamp."
        ),
    )
    parser.add_argument(
        "--bootstrap-state-retention-hours", type=int,
        help=(
            "When seeding from the complete causal panel, retain this many "
            "latest rows in bounded operator state for the subsequent "
            "incremental worker. This changes state retention only; it never "
            "truncates bootstrap feature computation."
        ),
    )
    parser.add_argument(
        "--emit-all-candidate-timestamps",
        action="store_true",
        help=(
            "Emit every complete-universe timestamp in the supplied "
            "chronological training chunk; live inference emits only the "
            "latest timestamp by default."
        ),
    )
    parser.add_argument(
        "--bootstrap-state", action="store_true",
        help="Seed exact rolling/transform state from the complete panel.",
    )
    parser.add_argument(
        "--raw-rolling-exact-seed-selector",
        action="append",
        default=[],
        help=(
            "Opt-in exact full-history seed for one raw rolling workset, in "
            "op:name:window form. Used only during --bootstrap-state so a "
            "known non-associative parent can continue exactly without "
            "broadening the normal incremental graph."
        ),
    )
    parser.add_argument(
        "--hybrid-exact-long-memory",
        action="store_true",
        help=(
            "Compute the parity-proven append-safe workset from state and only "
            "the audited long-memory residual workset from the sealed full panel."
        ),
    )
    parser.add_argument(
        "--debug-snapshot-dir",
        type=Path,
        help=(
            "Research-only latest-row snapshots before and after the generic "
            "causal transform. Never enable in a sealed inference bundle."
        ),
    )
    parser.add_argument(
        "--debug-snapshot-fields",
        help="Optional comma-separated feature names for debug snapshots.",
    )
    parser.add_argument(
        "--debug-full-history",
        action="store_true",
        help=(
            "Research-only: export complete requested parent histories at each "
            "semantic snapshot stage for one-time exact state bootstrap."
        ),
    )
    parser.add_argument(
        "--market-spectral-state",
        type=Path,
        help=(
            "Optional frozen market-spectral source contract. When omitted, "
            "the cache-local contract is created once and then reused."
        ),
    )
    parser.add_argument(
        "--stateful-exact-family",
        action="append",
        choices=(
            "price_memory", "residual_surprise", "cross_sectional", "final14",
            "orderbook_precomposite",
        ),
        default=[],
        help=(
            "Research promotion gate: replace this declared long-memory family "
            "with its exact persisted causal operator. Repeat for multiple "
            "families. Requires prior --bootstrap-state and exact parity."
        ),
    )
    parser.add_argument(
        "--expected-final14-contract-hash",
        help=(
            "Required frozen contract hash when the final14 family is enabled. "
            "The state is restored from cache-dir/strict_r3_final14.state."
        ),
    )
    parser.add_argument(
        "--expected-orderbook-precomposite-contract-hash",
        help=(
            "Required frozen contract hash when orderbook_precomposite is enabled. "
            "The state is restored from cache-dir/orderbook_feature_state.npz."
        ),
    )
    parser.add_argument("--side", choices=("long", "short"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.debug_full_history and args.debug_snapshot_dir is None:
        raise ValueError("--debug-full-history requires --debug-snapshot-dir")
    phase_started = time.perf_counter()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable incremental feature output exists: {args.out_dir}")
    canonical_cache_dir = args.cache_dir
    # All cache-producing modes are transactional, including the first
    # bootstrap and first immutable-bundle restore.  A process failure can
    # therefore leave only a private .txn.<pid> directory, never a partially
    # advanced canonical state.
    if args.cache_is_already_private:
        # The P8U warm worker first clones the ledger-selected cache into a
        # timestamped transaction directory, runs parity against a full causal
        # comparator, then atomically publishes only a small ledger pointer.
        # A second materializer-level clone doubled hourly state I/O without
        # adding any safety.  Direct mutation is safe *only* in that already
        # unpublished workspace; default callers retain the ordinary atomic
        # cache transaction below.
        state_transaction_cache = None
        state_transaction_cache_existed = False
    else:
        (
            args.cache_dir,
            state_transaction_cache,
            state_transaction_cache_existed,
        ) = _begin_state_transaction(canonical_cache_dir)
    phase_started = _mark_phase("state_transaction_copy", phase_started)
    restored_state_receipt = None
    frozen_causal_transform_order: list[str] = []
    pruned_causal_transform_states: list[str] = []
    if args.restore_state_bundle is not None:
        if args.cache_dir.exists():
            raise FileExistsError(
                "state restore requires a new cache directory: "
                f"{args.cache_dir}"
            )
        restored_state_receipt = restore_bundle(
            bundle_dir=args.restore_state_bundle,
            cache_dir=args.cache_dir,
            expected_contract_hash=args.expected_state_contract_hash,
        )
        frozen_causal_transform_order = _frozen_causal_transform_feature_order(
            args.restore_state_bundle
        )
        pruned_causal_transform_states = _prune_transitional_causal_transform_copies(
            args.cache_dir,
            expected_order=frozen_causal_transform_order,
        )
    phase_started = _mark_phase("optional_state_restore", phase_started)
    candidates = pd.read_parquet(args.candidates)
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True)
    candidates["__decision_ts__"] = pd.to_datetime(candidates["__decision_ts__"], utc=True)
    if not candidates["side_name"].astype(str).str.lower().eq(args.side).all():
        raise ValueError("incremental feature candidates have the wrong side")
    state = joblib.load(args.panel_state)
    phase_started = _mark_phase("candidate_and_panel_load", phase_started)
    source_panel_schema = _validate_panel_state_for_materialization(state)
    complete_panel = state["panel"]
    symbols = list(state["symbols"])
    latest_ts = candidates["__ts__"].max()
    # A recovered historical source bundle can contain bars later than the
    # decision being rebuilt.  The live producer normally never sees those
    # rows, but letting a replayed append state consume them would advance its
    # watermark past the requested decision and make the next state update
    # non-reproducible.  Trim *every* time-indexed source frame to the latest
    # signal timestamp before any cache/stateful constructor is opened.  The
    # signal-hour close itself remains available; only future observations are
    # excluded.  This is deliberately applied to ``complete_panel`` too,
    # because post-materialisation availability repairs must share the same
    # point-in-time view.
    complete_panel = {
        key: (
            value.loc[value.index <= latest_ts].copy()
            if isinstance(value, pd.DataFrame)
            and isinstance(value.index, pd.DatetimeIndex)
            else value
        )
        for key, value in complete_panel.items()
    }
    panel = complete_panel
    current = (
        candidates.copy()
        if args.emit_all_candidate_timestamps
        else candidates.loc[candidates["__ts__"].eq(latest_ts)].copy()
    )
    for ts, group in current.groupby("__ts__", sort=False):
        if set(group["__symbol__"].astype(str)) != set(symbols):
            raise ValueError(
                f"feature timestamp is not the complete frozen universe: {ts}"
            )
    if args.stateful_tail_hours is not None and args.stateful_tail_hours < 72:
        raise ValueError("stateful feature tail must retain at least 72 causal hours")
    if (
        args.bootstrap_state_retention_hours is not None
        and args.bootstrap_state_retention_hours < 72
    ):
        raise ValueError("bootstrap state retention must retain at least 72 causal hours")
    if args.bootstrap_state and args.stateful_tail_hours is not None:
        raise ValueError("state bootstrap must consume the complete causal panel")
    if args.bootstrap_state_retention_hours is not None and not args.bootstrap_state:
        raise ValueError("bootstrap state retention requires --bootstrap-state")
    if args.hybrid_exact_long_memory and args.stateful_tail_hours is None:
        raise ValueError("hybrid exact mode requires --stateful-tail-hours")
    if args.stateful_tail_hours is not None:
        tail_start = latest_ts - pd.Timedelta(hours=args.stateful_tail_hours)
        panel = {
            key: (
                value.loc[value.index >= tail_start].copy()
                if isinstance(value, pd.DataFrame) else value
            )
            for key, value in panel.items()
        }
        # The canonical order-book adapter applies an unbounded causal ffill
        # before its one-bar shift.  Preserve exactly the last observation
        # available before the bounded tail by seeding only the first retained
        # row.  This is operator state, not an approximation or future fill.
        for name, value in list(panel.items()):
            if not str(name).startswith("orderbook_") or not isinstance(value, pd.DataFrame):
                continue
            complete = complete_panel.get(name)
            if not isinstance(complete, pd.DataFrame) or value.empty:
                continue
            before = complete.loc[complete.index < value.index[0]]
            if before.empty:
                continue
            carry = before.ffill().iloc[-1]
            first = value.iloc[0].copy()
            missing = first.isna() & carry.notna()
            if bool(missing.any()):
                first.loc[missing] = carry.loc[missing]
                value.iloc[0] = first
                panel[name] = value
    phase_started = _mark_phase("bounded_panel_construction", phase_started)

    requested_feature_plan_sha256 = None
    if args.requested_features_json is not None:
        plan_payload = json.loads(args.requested_features_json.read_text())
        plan_features = (
            plan_payload.get("full_union")
            if isinstance(plan_payload, dict) else plan_payload
        )
        if (
            not isinstance(plan_features, list)
            or not plan_features
            or not all(isinstance(name, str) and name for name in plan_features)
            or len(set(plan_features)) != len(plan_features)
        ):
            raise ValueError(
                "--requested-features-json must be a list or object with a "
                "non-empty, duplicate-free full_union list"
            )
        requested = list(plan_features)
        import hashlib
        requested_feature_plan_sha256 = hashlib.sha256(
            args.requested_features_json.read_bytes()
        ).hexdigest()
    else:
        base_contract = _load_contract()
        declared = json.loads(
            (ROOT / "config/strict_r3_canonical_v2_feature_contract.json").read_text()
        )
        requested = list(dict.fromkeys([
            *base_contract[args.side], *declared["severe_context_fields"],
            *SESSION_CALENDAR_BASE_CANDIDATE_FEATURE_KEYS,
            *FROZEN_GENERATION_DEPENDENCIES,
        ]))
    # Session-calendar values are deterministic decision-time helpers.  They
    # are intentionally available to the compute graph, but no frozen strict
    # R3 model consumes them.  Do not let a later convenience addition widen
    # the persisted canonical model matrix: its schema is part of the live
    # append-state contract and must remain byte-compatible with the verified
    # predecessor bundles.
    # A sealed explicit plan owns its exact output schema.  P8U's 175-field
    # union includes ``hour_of_week_sin`` as a deterministic decision-time
    # input, so suppressing calendar fields here silently produced 174 rather
    # than 175 columns and made a state-parity proof impossible.  Preserve the
    # historic default exclusion only when no explicit plan has been supplied
    # by a versioned contract.
    output_requested = (
        list(requested)
        if args.requested_features_json is not None
        else [
            name for name in requested
            if name not in SESSION_CALENDAR_BASE_CANDIDATE_FEATURE_KEYS
        ]
    )
    spectral_state_path = Path(
        args.market_spectral_state
        or (args.cache_dir / "market_spectral_source_state.json")
    )
    spectral_parent_keys = _frozen_spectral_parent_keys(spectral_state_path)
    compute_requested = list(dict.fromkeys([*requested, *spectral_parent_keys]))
    stateful_exact_families = frozenset(args.stateful_exact_family)
    if "final14" in stateful_exact_families and not args.expected_final14_contract_hash:
        raise ValueError(
            "stateful final14 requires --expected-final14-contract-hash"
        )
    if (
        "orderbook_precomposite" in stateful_exact_families
        and not args.expected_orderbook_precomposite_contract_hash
    ):
        raise ValueError(
            "stateful orderbook_precomposite requires "
            "--expected-orderbook-precomposite-contract-hash"
        )
    state_replaced_fields: set[str] = set()
    if "price_memory" in stateful_exact_families:
        state_replaced_fields.update(PRICE_MEMORY_PROMOTED_FIELDS)
    if "residual_surprise" in stateful_exact_families:
        state_replaced_fields.update(RESIDUAL_SURPRISE_STATE_FIELDS)
    if "cross_sectional" in stateful_exact_families:
        state_replaced_fields.update(CROSS_SECTIONAL_STATE_FIELDS)
    if "final14" in stateful_exact_families:
        state_replaced_fields.update(FINAL14_FIELD_ORDER)
    if "orderbook_precomposite" in stateful_exact_families:
        state_replaced_fields.update(ORDERBOOK_OUTPUTS)
    # Direct-state contracts are frozen as complete families. Request every
    # member even when only a subset reaches the deployed 120-field matrix so
    # transform-state identity cannot change with a model workset.
    compute_requested = list(dict.fromkeys([
        *compute_requested,
        *(
            PRICE_MEMORY_STATE_FIELD_ORDER
            if "price_memory" in stateful_exact_families else ()
        ),
        *(
            (
                "ret1h", "excess_6h", "spike_score", "grind_score",
                "volume_price_corr_10h", "ret4h", "log_quote_volume",
                "ob_spread_bps", "bars_in_high_vol_state_log_norm",
                "ret48h_bench_resid",
                *tuple(
                    str(value)
                    for value in (
                        CFG.get("MODEL_REGIME_COMPOSITE_EIGEN_GROUPS", {}) or {}
                    ).get("open_interest", [])
                ),
            )
            if (
                "final14" in stateful_exact_families
                or bool(args.debug_full_history)
            ) else ()
        ),
        *sorted(state_replaced_fields),
    ]))
    cfg = dict(CFG)
    # The global configuration can carry a training/live parity source.  It is
    # deliberately not a writable state namespace for this challenger.
    cfg.pop("training_live_parity_contract", None)
    cfg.update({
        "atr_n": 14,
        "use_perps": True,
        "feature_portability_mode": "off",
        "feature_portability_strict": False,
        "live_raw_feature_compute_preserve_portability_mode": True,
        "enable_orderbook_features": False,
        "enable_orderbook_wall_features": False,
        # The shared canonical batch computation has already materialized the
        # frozen order-book fields.  The generic live wrapper's alternate
        # summary synthesizer uses different shifting/fill semantics and must
        # not overwrite them in a training-parity challenger.
        "live_materialize_orderbook_model_features": False,
        "live_lgbm_mask_feature_fast_path_enabled": False,
        # Preserve the historical complete regime workset.  Although the
        # strict-R3 model matrix does not directly consume every helper, the
        # OI cross-sectional composites depend on the complete causal family.
        # Skipping it changes their values and breaks the sealed live contract.
        "feature_skip_unrequested_regime_block": False,
        "live_feature_cache_namespace": (
            str(args.feature_cache_namespace)
            if args.feature_cache_namespace
            else "strict_r3_schema_v13_canonical120"
        ),
        "live_feature_snapshot_cache_dir": str(args.cache_dir / "feature_cache"),
        "live_feature_snapshot_cache_enabled": True,
        "live_feature_rolling_cache_enabled": True,
        "live_feature_rolling_cache_latest_only_read_enabled": True,
        "live_feature_latest_row_incremental_enabled": True,
        "live_feature_return_latest_only": True,
        "live_feature_persist_after_scoring": False,
        "live_model_feature_tail_recompute_enabled": True,
        # Bootstrap from the canonical batch definitions.  Enabling an empty
        # rolling state on a full historical request seeds only the transform
        # tail and is not numerically equivalent to the frozen batch contract.
        # The append cache below remains incremental; stateful rolling kernels
        # may only be enabled after an explicit canonical-state seeding audit.
        "live_raw_rolling_state_enabled": bool(
            args.bootstrap_state or args.stateful_tail_hours is not None
        ),
        # An EWMA's complete recurrence is observable on the first append.
        # Seed its accumulator over the full causal bootstrap panel; resumed
        # calls merely load and extend it.  This is required for feature
        # parity, not an optimisation or a new source of information.
        "live_raw_rolling_state_exact_accumulator_seed": bool(
            args.bootstrap_state
        ),
        "live_raw_rolling_state_path": str(args.cache_dir / "raw_rolling_state.npz"),
        "live_raw_rolling_state_container_enabled": False,
        "live_raw_rolling_state_container_path": str(args.cache_dir / "raw_rolling_state.sqlite"),
        "live_raw_rolling_state_sparse_prefix_enabled": True,
        "live_causal_transform_state_enabled": bool(
            args.bootstrap_state or args.stateful_tail_hours is not None
        ),
        # The batch causal normaliser retains a shifted numerical anchor even
        # after the observation which established it has left the trailing
        # window.  A tail-only bootstrap therefore diverges on its first
        # append for sparse or flat histories.  Replay the supplied causal
        # bootstrap history once to seed that compact sufficient state.  This
        # applies only to a cold --bootstrap-state run; ordinary hourly
        # appends still reopen and extend the persisted bounded state.
        "feature_causal_transform_exact_full_state_seed": bool(
            args.bootstrap_state
        ),
        "live_causal_transform_exact_full_state_seed": bool(
            args.bootstrap_state
        ),
        "live_causal_transform_state_path": str(args.cache_dir / "causal_transform_state.npz"),
        "live_market_spectral_state_path": str(spectral_state_path),
        "live_market_spectral_history_state_enabled": bool(
            args.bootstrap_state or args.stateful_tail_hours is not None
        ),
        "live_derived_history_state_enabled": bool(
            args.bootstrap_state or args.stateful_tail_hours is not None
        ),
        "live_derived_history_state_dir": str(args.cache_dir / "derived_history"),
        "live_nested_derived_state_enabled": bool(
            args.bootstrap_state or args.stateful_tail_hours is not None
        ),
        "live_nested_derived_state_path": str(
            args.cache_dir / "nested_derived_feature_state.sqlite"
        ),
        # This 14-day window is counted within each hour-of-day bucket: its
        # uncompressed parent history spans up to 336 calendar days.  Persist
        # the compact grouped ring rather than silently approximating it from
        # a bounded hourly source tail.
        "live_grouped_rolling_state_enabled": bool(
            args.bootstrap_state or args.stateful_tail_hours is not None
        ),
        "live_grouped_rolling_state_path": str(
            args.cache_dir / "grouped_rolling_state.npz"
        ),
        "live_grouped_rolling_state_scope": (
            str(args.feature_cache_namespace)
            if args.feature_cache_namespace
            else "strict_r3_schema_v13_canonical120"
        ),
        # ``atr_compression_ratio`` has a long recursive ATR parent that is
        # distinct from the raw-roll registry. Retain its exact recurrent
        # state under the same hash-bound cache namespace.
        "live_ewma_state_enabled": bool(
            args.bootstrap_state or args.stateful_tail_hours is not None
        ),
        "live_ewma_state_dir": str(args.cache_dir / "ewma_state"),
        "live_ewma_state_scope": (
            str(args.feature_cache_namespace)
            if args.feature_cache_namespace
            else "strict_r3_schema_v13_canonical120"
        ),
        # Calendar/session fields are deterministic at the decision time and
        # do not feed any nested derived calculation.  Exclude only this
        # stateless family from the persisted nested-state contract so adding
        # them to a candidate matrix cannot invalidate an otherwise identical
        # causal operator state.
        "live_nested_derived_state_feature_keys": sorted(
            set(compute_requested).difference(
                set(SESSION_CALENDAR_BASE_CANDIDATE_FEATURE_KEYS)
            ).difference(NESTED_DERIVED_RAW_PRIMITIVE_KEYS)
        ),
        "live_price_memory_state_path": str(
            args.cache_dir / "price_memory_feature_state.npz"
        ),
        "live_primitive_price_state_path": str(
            args.cache_dir / "primitive_price_state.npz"
        ),
        "live_cross_sectional_composite_state_path": str(
            args.cache_dir / "cross_sectional_composite_state.json"
        ),
        "live_price_memory_pipeline_state_enabled": bool(
            "price_memory" in stateful_exact_families
        ),
        "live_price_memory_pipeline_state_dir": str(
            args.cache_dir / "price_memory_pipeline"
        ),
        "live_residual_surprise_state_enabled": bool(
            "residual_surprise" in stateful_exact_families
        ),
        "live_residual_surprise_state_path": str(
            args.cache_dir / "residual_surprise_state.npz"
        ),
        "live_cross_sectional_composite_state_enabled": bool(
            "cross_sectional" in stateful_exact_families
        ),
        "live_direct_causal_output_state_enabled": bool(
            "price_memory" in stateful_exact_families
        ),
        "live_direct_causal_output_state_path": str(
            args.cache_dir / "price_memory_pipeline" /
            "direct_causal_output_state.json"
        ),
        "live_direct_causal_output_state_keys": list(
            PRICE_MEMORY_STATE_FIELD_ORDER
        ) if "price_memory" in stateful_exact_families else [],
        "live_strict_r3_final14_state_enabled": bool(
            "final14" in stateful_exact_families
        ),
        "live_strict_r3_final14_state_path": str(
            args.cache_dir / "strict_r3_final14.state"
        ),
        "live_strict_r3_final14_contract_hash": (
            args.expected_final14_contract_hash
            if "final14" in stateful_exact_families else None
        ),
        "live_orderbook_precomposite_state_enabled": bool(
            "orderbook_precomposite" in stateful_exact_families
        ),
        "live_orderbook_precomposite_state_path": str(
            args.cache_dir / "orderbook_feature_state.npz"
        ),
        "live_orderbook_precomposite_contract_hash": (
            args.expected_orderbook_precomposite_contract_hash
            if "orderbook_precomposite" in stateful_exact_families else None
        ),
        # The nested-derived overlay must retain the same causal horizon as
        # the caller's append state.  A bootstrap previously hard-coded 1536
        # rows while a resumed worker could consume a longer source tail,
        # causing valid state metadata to reopen but overlay an incomplete
        # parent history.  The sealed minimum remains 1536 for legacy/full
        # materializations; a declared stateful horizon can only lengthen it.
        "feature_nested_derived_state_max_rows": max(
            1536,
            int(
                args.bootstrap_state_retention_hours or 0
                if args.bootstrap_state else args.stateful_tail_hours or 0
            ),
        ),
        "live_oi_long_iqr_state_dir": str(args.cache_dir / "oi_long_iqr"),
        "live_fixed_ffd_state_enabled": bool(
            args.bootstrap_state or args.stateful_tail_hours is not None
        ),
        "live_fixed_ffd_state_dir": str(args.cache_dir / "fixed_ffd"),
        # The resumed feature graph consumes ``stateful_tail_hours`` of
        # causal parents.  Fixed FFD is a transformed *parent*, rather than
        # merely a latest-row feature: retaining only its old 768-row display
        # tail leaves the earlier part of a 2,304-hour resumed graph as NaN.
        # That did not change the FFD append value itself, but it changed
        # downstream rolling descendants (trend/path/volatility) at the
        # first post-bootstrap append.  Retain exactly the same bounded
        # horizon the graph will consume, so every descendant sees the
        # canonical transformed parent throughout its local window.  This is
        # cached state, not a full-history runtime fallback.
        "feature_fixed_ffd_output_history_rows": max(
            768,
            int(
                args.bootstrap_state_retention_hours or 0
                if args.bootstrap_state else args.stateful_tail_hours or 0
            ),
        ),
        "feature_derived_history_max_rows": 1536,
        "feature_raw_rolling_state_exact_seed_selectors": list(
            args.raw_rolling_exact_seed_selector or []
        ),
        "static_feature_store_write_enabled": False,
        "run_id": "strict_r3_schema_v13_canonical120",
        "live_feature_source_run_ids": ["strict_r3_schema_v13_canonical120"],
        "live_static_feature_store_id_override": "strict_r3_schema_v13_canonical120",
        "feature_raw_rolling_state_scope": "strict_r3_schema_v13_canonical120",
        "feature_causal_transform_state_scope": "strict_r3_schema_v13_canonical120",
        "live_causal_transform_expected_feature_order": (
            frozen_causal_transform_order
        ),
    })
    if args.debug_snapshot_dir is not None:
        cfg["feature_state_debug_snapshot_dir"] = str(args.debug_snapshot_dir)
        cfg["feature_state_debug_snapshot_full_history"] = bool(
            args.debug_full_history
        )
        cfg["feature_state_debug_spectral_history_rows"] = 97
        cfg["feature_state_debug_snapshot_fields"] = [
            value.strip()
            for value in str(args.debug_snapshot_fields or "").split(",")
            if value.strip()
        ] or compute_requested
    # Use only canonical constructors. In hybrid mode the ordinary workset is
    # evaluated on the append-state tail and the audited residual workset is
    # replaced with values computed on the immutable complete panel.
    short_requested = (
        [
            name for name in compute_requested
            if (
                name not in EXACT_LONG_MEMORY_FIELDS
                or name in spectral_parent_keys
                or name in state_replaced_fields
                # Direct transform state is an atomic family contract.  Raw
                # members that are not yet promoted must still be computed so
                # the state can advance on one common watermark; the exact
                # fallback below remains authoritative for those fields.
                or (
                    "price_memory" in stateful_exact_families
                    and name in PRICE_MEMORY_STATE_FIELDS
                )
            )
        ]
        if args.hybrid_exact_long_memory else compute_requested
    )
    if args.hybrid_exact_long_memory:
        # The hybrid short workset has a deliberately different dependency
        # contract from the canonical state bootstrap. Never open the latter
        # under a reduced key set. Exact full-history values replace the
        # declared long-memory fields below; this mode is a golden comparator,
        # not the promoted incremental producer.
        for key in (
            "live_nested_derived_state_enabled",
            "live_derived_history_state_enabled",
            "live_market_spectral_history_state_enabled",
            "live_raw_rolling_state_enabled",
            "live_causal_transform_state_enabled",
            "live_fixed_ffd_state_enabled",
        ):
            cfg[key] = False
    features, feature_index, feature_columns = _compute_contract_features(
        panel,
        symbols=symbols,
        requested=short_requested,
        cfg=cfg,
        # Keep the bounded 185-field graph fast.  Only the inexpensive shared
        # market-gate parent receives complete causal history so its cumulative
        # index/ATR-selector recurrence is bit-identical at a restart boundary.
        market_gate_panel=(
            complete_panel if args.stateful_tail_hours is not None else None
        ),
        # ``volatility_zscore`` is one terminal 30-day robust-z field.  Its
        # exact parent is inexpensive to derive from complete causal OHLC and
        # avoids a ULP-level restart drift without widening the feature graph.
        volatility_zscore_panel=(
            complete_panel if args.stateful_tail_hours is not None else None
        ),
    )
    phase_started = _mark_phase("bounded_feature_graph", phase_started)
    exact_long_memory_fields: list[str] = []
    if args.hybrid_exact_long_memory:
        # Recovery must never publish a mixed semantic matrix.  The bounded
        # append graph is useful to advance/cache causal operators, but not
        # every deployed derived family has a persisted final-output state
        # yet.  Replacing only the known long-memory subset left direct-state
        # and cross-sectional descendants numerically different from the
        # canonical batch feature graph.  In this explicit recovery mode,
        # compute the complete deployed output contract on the full causal
        # panel and use it as the authoritative matrix.  State still advances
        # above for continuity; it simply cannot lower the fidelity of a
        # no-order historical reconstruction.
        exact_long_memory_fields = list(output_requested)
        # Full-history fallback must be read-only: it cannot advance the append
        # state a second time for the same timestamp.
        exact_cfg = dict(cfg)
        exact_cfg["live_raw_rolling_state_enabled"] = False
        exact_cfg["live_causal_transform_state_enabled"] = False
        # The read-only fallback has a deliberately smaller feature workset
        # than the append graph. Never open append-state namespaces under that
        # incompatible key contract or advance them a second time.
        exact_cfg["live_price_memory_pipeline_state_enabled"] = False
        exact_cfg["live_residual_surprise_state_enabled"] = False
        exact_cfg["live_cross_sectional_composite_state_enabled"] = False
        exact_cfg["live_direct_causal_output_state_enabled"] = False
        exact_cfg["feature_direct_causal_output_state_enabled"] = False
        # A frozen spectral output is an exact long-memory field, but its
        # selected primitive parents are not necessarily model columns.  The
        # fallback must nevertheless compute them, otherwise the frozen
        # spectral source contract cannot be reconstructed at a recovered
        # boundary.  They remain helper-only and are never copied into the
        # persisted candidate matrix below.
        exact_compute_requested = list(dict.fromkeys([
            *compute_requested,
            *spectral_parent_keys,
        ]))
        exact_features, exact_index, exact_columns = _compute_contract_features(
            complete_panel,
            symbols=symbols,
            requested=exact_compute_requested,
            cfg=exact_cfg,
        )
        if not exact_index.equals(pd.Index(complete_panel["close"].index)):
            raise AssertionError("long-memory fallback changed the canonical time index")
        if list(exact_columns) != list(symbols):
            raise AssertionError("long-memory fallback changed the canonical symbol order")
        for name in exact_long_memory_fields:
            frame = exact_features.get(name)
            if not isinstance(frame, pd.DataFrame):
                raise KeyError(f"long-memory fallback did not produce {name}")
            features[name] = frame
    phase_started = _mark_phase("exact_long_memory_fallback", phase_started)
    # A full historical bootstrap has already materialised the entire feature
    # graph.  Constructing its 170-symbol output matrix in one DataFrame used
    # to add a second multi-gigabyte peak and could be OOM-killed.  Write the
    # exact same candidate-order matrix in chronological batches instead.
    # Each batch uses the identical vectorised lookup and source-missingness
    # contract, and the resulting Parquet row order remains timestamp then ID.
    current = current.sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    args.out_dir.mkdir(parents=True)
    path = args.out_dir / "canonical120_features.parquet"
    stream_output = (
        args.feature_prefix is None
        and int(current["__ts__"].nunique()) > 1
    )
    if stream_output:
        writer: pq.ParquetWriter | None = None
        row_count = 0
        unique_timestamps = pd.DatetimeIndex(
            current["__ts__"].drop_duplicates().sort_values()
        )
        # 72 hourly universes is small enough to remain well below the former
        # peak while retaining efficient vectorised field gathers.
        for start_idx in range(0, len(unique_timestamps), 72):
            batch_timestamps = unique_timestamps[start_idx:start_idx + 72]
            batch_candidates = current.loc[current["__ts__"].isin(batch_timestamps)]
            latest = _latest_matrix(
                features, candidates=batch_candidates, requested=output_requested
            )
            _apply_current_source_missingness(
                latest, complete_panel=complete_panel, cfg=cfg
            )
            table = pa.Table.from_pandas(latest, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            row_count += len(latest)
            del latest, table
        if writer is None or row_count != len(current):
            if writer is not None:
                writer.close()
            raise AssertionError("bounded feature materialisation lost candidate rows")
        writer.close()
        output_rows = row_count
    else:
        latest = _latest_matrix(
            features, candidates=current, requested=output_requested
        )
        _apply_current_source_missingness(
            latest, complete_panel=complete_panel, cfg=cfg
        )
        if args.feature_prefix is not None:
            prefix = pd.read_parquet(args.feature_prefix)
            prefix["candidate_id"] = prefix["candidate_id"].astype(str)
            if set(prefix["candidate_id"]).intersection(latest["candidate_id"].astype(str)):
                raise ValueError("incremental feature append overlaps immutable candidate IDs")
            columns = list(prefix.columns) + [c for c in latest.columns if c not in prefix]
            output = pd.concat([
                prefix.reindex(columns=columns), latest.reindex(columns=columns),
            ], ignore_index=True)
        else:
            output = latest
        output = output.sort_values(["__decision_ts__", "candidate_id"], kind="stable")
        output.to_parquet(path, index=False, compression="zstd")
        output_rows = len(output)
        del output, latest
    # The post-materialisation repair is intentionally retained; release the
    # high-dimensional graph first so its read/modify/write step has bounded
    # memory even for a full historical bootstrap.
    del features
    gc.collect()
    phase_started = _mark_phase("latest_matrix_and_missingness", phase_started)
    phase_started = _mark_phase("feature_output_write", phase_started)
    # Use the canonical causal post-materialisation definitions.  They operate
    # on the complete point-in-time universe and repair fields whose parents
    # become available only after the main feature call.
    _repair_cross_asset_state_fields(
        path,
        candidates=current,
        # The state panel is deliberately a bounded causal tail, whereas
        # ``history_start`` names the original bootstrap history.  The repair
        # consumes this already bounded panel directly and only needs its
        # actual earliest timestamp for its own rolling definitions.  Passing
        # the bootstrap start makes a valid recovered tail fail its causality
        # assertion despite containing no future data.
        start=pd.DatetimeIndex(complete_panel["close"].index).min(),
        end=pd.Timestamp(state["end_exclusive"]),
        source_panel=complete_panel,
    )
    phase_started = _mark_phase("post_materialisation_repair", phase_started)
    _refresh_feature_coverage(path, args.out_dir / "feature_coverage.parquet")
    phase_started = _mark_phase("coverage_audit", phase_started)
    manifest = {
        "schema": "strict_r3_forward_feature_incremental_v13_challenger",
        "panel_state": str(args.panel_state),
        "panel_state_schema": source_panel_schema,
        "feature_prefix": str(args.feature_prefix) if args.feature_prefix else None,
        "latest_signal_ts": latest_ts.isoformat(),
        "new_rows": int(len(current)),
        "output_rows": int(output_rows),
        "requested_fields": int(len(output_requested)),
        "requested_features_json": (
            str(args.requested_features_json)
            if args.requested_features_json is not None else None
        ),
        "requested_features_json_sha256": requested_feature_plan_sha256,
        "computed_helper_fields": int(
            len(requested) - len(output_requested)
        ),
        "excluded_stateless_output_fields": sorted(
            set(requested).intersection(SESSION_CALENDAR_BASE_CANDIDATE_FEATURE_KEYS)
        ),
        "feature_cache_namespace": cfg["live_feature_cache_namespace"],
        "cache_is_already_private": bool(args.cache_is_already_private),
        "outcome_columns_consumed": [],
        "state_bootstrap": bool(args.bootstrap_state),
        "stateful_tail_hours": args.stateful_tail_hours,
        "bootstrap_state_retention_hours": args.bootstrap_state_retention_hours,
        "emit_all_candidate_timestamps": bool(args.emit_all_candidate_timestamps),
        "hybrid_exact_long_memory": bool(args.hybrid_exact_long_memory),
        "exact_long_memory_fields": exact_long_memory_fields,
        "stateful_exact_families": sorted(stateful_exact_families),
        "strict_r3_final14_state": (
            str(canonical_cache_dir / "strict_r3_final14.state")
            if "final14" in stateful_exact_families else None
        ),
        "strict_r3_final14_contract_hash": (
            args.expected_final14_contract_hash
            if "final14" in stateful_exact_families else None
        ),
        "state_replaced_long_memory_fields": sorted(
            EXACT_LONG_MEMORY_FIELDS.intersection(requested).intersection(
                state_replaced_fields
            )
        ),
        "debug_snapshot_dir": (
            str(args.debug_snapshot_dir) if args.debug_snapshot_dir else None
        ),
        "debug_full_history": bool(args.debug_full_history),
        "market_spectral_state": str(
            (
                canonical_cache_dir / "market_spectral_source_state.json"
                if args.market_spectral_state is None
                else spectral_state_path
            )
        ),
        "spectral_dependency_fields": spectral_parent_keys,
        "restored_state_bundle": (
            str(args.restore_state_bundle)
            if args.restore_state_bundle is not None else None
        ),
        "state_restore_receipt": (
            {
                **restored_state_receipt,
                "cache_dir": str(canonical_cache_dir),
            }
            if restored_state_receipt is not None else None
        ),
        "pruned_transitional_causal_transform_states": (
            pruned_causal_transform_states
        ),
        "phase_runtime_seconds": phase_timings,
        "runtime_seconds_before_state_commit": float(
            time.perf_counter() - run_started
        ),
    }
    (args.out_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if state_transaction_cache is not None:
        _commit_state_transaction(
            args.cache_dir,
            state_transaction_cache,
            state_transaction_cache_existed,
        )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
