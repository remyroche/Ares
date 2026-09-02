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
import json
import os
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG  # noqa: E402
from extreme_price_movements.features import (  # noqa: E402
    _apply_derived_feature_history_state,
    _add_regime_panel_composite_features,
    _regime_composite_group_from_key,
    _regime_composite_parent_from_key,
    add_regime_gates,
    compute_features_hourly,
    compute_market_features,
)
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
    "negative_breadth_pct",
    "ob_depth_l20_to_qv_z_7d",
    "ob_spread_bps",
    "ob_spread_bps_z_24h",
    "ob_spread_z_24h",
    "ob_trade_size_to_l1_depth_z_24h",
    "prior_volatility",
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
    for field in requested:
        frame = features.get(field)
        if not isinstance(frame, pd.DataFrame):
            value_columns[field] = np.full(len(identities), np.nan, dtype=np.float32)
            continue
        values = np.full(len(identities), np.nan, dtype=np.float32)
        for ts in timestamps.unique():
            positions = np.flatnonzero(timestamps.eq(ts).to_numpy())
            if ts not in frame.index:
                continue
            row = frame.loc[ts]
            values[positions] = pd.to_numeric(
                row.reindex(symbols.iloc[positions].to_list()), errors="coerce"
            ).to_numpy(dtype=np.float32, copy=False)
        value_columns[field] = values
    return pd.concat(
        [identities.reset_index(drop=True), pd.DataFrame(value_columns)], axis=1
    )


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
) -> tuple[dict[str, pd.DataFrame], pd.Index, pd.Index]:
    """Run the canonical constructors for one declared feature workset."""
    market = compute_market_features(panel, symbols)
    gates = add_regime_gates(
        market, gate_vol_lookback_hours=24 * 7, gate_trend_thr=0.0,
    )
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
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError("unsupported source-panel state")
    complete_panel = state["panel"]
    panel = complete_panel
    symbols = list(state["symbols"])
    latest_ts = candidates["__ts__"].max()
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
    if args.bootstrap_state and args.stateful_tail_hours is not None:
        raise ValueError("state bootstrap must consume the complete causal panel")
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

    base_contract = _load_contract()
    declared = json.loads(
        (ROOT / "config/strict_r3_canonical_v2_feature_contract.json").read_text()
    )
    requested = list(dict.fromkeys([
        *base_contract[args.side], *declared["severe_context_fields"],
        *FROZEN_GENERATION_DEPENDENCIES,
    ]))
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
        "live_feature_cache_namespace": "strict_r3_schema_v13_canonical120",
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
        "live_raw_rolling_state_path": str(args.cache_dir / "raw_rolling_state.npz"),
        "live_raw_rolling_state_container_enabled": False,
        "live_raw_rolling_state_container_path": str(args.cache_dir / "raw_rolling_state.sqlite"),
        "live_raw_rolling_state_sparse_prefix_enabled": True,
        "live_causal_transform_state_enabled": bool(
            args.bootstrap_state or args.stateful_tail_hours is not None
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
        "feature_nested_derived_state_max_rows": 1536,
        "live_oi_long_iqr_state_dir": str(args.cache_dir / "oi_long_iqr"),
        "live_fixed_ffd_state_enabled": bool(
            args.bootstrap_state or args.stateful_tail_hours is not None
        ),
        "live_fixed_ffd_state_dir": str(args.cache_dir / "fixed_ffd"),
        "feature_fixed_ffd_output_history_rows": 768,
        "feature_derived_history_max_rows": 1536,
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
        panel, symbols=symbols, requested=short_requested, cfg=cfg,
    )
    phase_started = _mark_phase("bounded_feature_graph", phase_started)
    exact_long_memory_fields: list[str] = []
    if args.hybrid_exact_long_memory:
        exact_long_memory_fields = sorted(
            EXACT_LONG_MEMORY_FIELDS.intersection(requested).difference(
                state_replaced_fields
            )
        )
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
        exact_features, exact_index, exact_columns = _compute_contract_features(
            complete_panel,
            symbols=symbols,
            requested=exact_long_memory_fields,
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
    latest = _latest_matrix(features, candidates=current, requested=requested)
    # Current source absence must stay missing until the authorised
    # same-timestamp trade-size fallback runs below. A stateful rolling kernel
    # must never reinterpret an unavailable book as an economically neutral 0.
    bid = complete_panel.get("orderbook_best_bid")
    ask = complete_panel.get("orderbook_best_ask")
    if isinstance(bid, pd.DataFrame) and isinstance(ask, pd.DataFrame):
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
    phase_started = _mark_phase("latest_matrix_and_missingness", phase_started)
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
    args.out_dir.mkdir(parents=True)
    path = args.out_dir / "canonical120_features.parquet"
    output.to_parquet(path, index=False, compression="zstd")
    phase_started = _mark_phase("feature_output_write", phase_started)
    # Use the canonical causal post-materialisation definitions.  They operate
    # on the complete point-in-time universe and repair fields whose parents
    # become available only after the main feature call.
    _repair_cross_asset_state_fields(
        path,
        candidates=current,
        start=pd.Timestamp(state["history_start"]),
        end=pd.Timestamp(state["end_exclusive"]),
        source_panel=complete_panel,
    )
    phase_started = _mark_phase("post_materialisation_repair", phase_started)
    _refresh_feature_coverage(path, args.out_dir / "feature_coverage.parquet")
    phase_started = _mark_phase("coverage_audit", phase_started)
    manifest = {
        "schema": "strict_r3_forward_feature_incremental_v13_challenger",
        "panel_state": str(args.panel_state),
        "feature_prefix": str(args.feature_prefix) if args.feature_prefix else None,
        "latest_signal_ts": latest_ts.isoformat(),
        "new_rows": int(len(latest)),
        "output_rows": int(len(output)),
        "requested_fields": int(len(requested)),
        "feature_cache_namespace": cfg["live_feature_cache_namespace"],
        "outcome_columns_consumed": [],
        "state_bootstrap": bool(args.bootstrap_state),
        "stateful_tail_hours": args.stateful_tail_hours,
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
    _commit_state_transaction(
        args.cache_dir,
        state_transaction_cache,
        state_transaction_cache_existed,
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
