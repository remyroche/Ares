#!/usr/bin/env python3
"""
CLI entry point for extreme_price_movements pipeline.

Usage:
    python3 extreme_price_movements/run_pipeline.py labels
"""
import os
import sys
import warnings

# Avoid expensive/warning-prone Matplotlib cache initialization under read-only HOME.
_mpl_cfg = os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_epm")
os.environ.setdefault("MPLBACKEND", "Agg")
_loky_cpu = str(os.environ.get("LOKY_MAX_CPU_COUNT", "")).strip()
if not _loky_cpu.isdigit():
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores for the following reason:",
    category=UserWarning,
)
try:
    os.makedirs(_mpl_cfg, exist_ok=True)
except Exception:
    pass

# Add parent directory to Python path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from extreme_price_movements.config import CFG, enable_perp_feature_keys
from extreme_price_movements.data_store import (
    PartitionedOHLCVStore,
    make_perp_exchange,
    make_spot_exchange,
)
from extreme_price_movements.offline_optimisers.params_store import (
    apply_offline_optimizer_best_params,
)
import extreme_price_movements.mask_optimiser as mask_opt
from extreme_price_movements.optimise import (
    Policy,
    run_optimise_from_ridge_oof,
    run_optimise_step,
)
from extreme_price_movements.pipeline_steps import (
    run_backtest_step,
    run_feature_generation_step,
    run_label_generation_step_v2,
    run_risk_optimization_step,
    run_sizer_step,
    run_training_step,
)
from extreme_price_movements.universe import (
    build_fetch_universe,
    refresh_margin_universe_daily,
)
from extreme_price_movements.utils import tprint

# SINGLE SOURCE OF TRUTH FOR FEES - All fee configuration comes from these constants
# Spot trading fees (default)
BASE_ROUND_TRIP_FEE_PCT = 0.3  # 0.3% round-trip = 0.15% per side (15 bps)
# Perpetual trading fees (when --perps flag used)
PERP_ROUND_TRIP_FEE_PCT = 0.1  # 0.1% round-trip = 0.05% per side (5 bps)

# Market order fee per side (used when not using limit orders)
MARKET_ORDER_FEE_BPS = 25.0  # 0.25% per side
# Limit order fee per side (used when limit order fills)
LIMIT_ORDER_FEE_BPS = 10.0  # 0.10% per side


def _apply_fee_model(cfg: dict, round_trip_fee_pct: float) -> None:
    """Normalize fee keys used across training, sizing, and optimisation steps."""
    rt = float(round_trip_fee_pct)
    side_bps = rt * 100.0 / 2.0
    fee_dec = rt / 100.0
    cfg["label_round_trip_fee_pct"] = rt
    cfg["sample_weight_fee_rt"] = fee_dec
    cfg["fee_bps"] = side_bps
    cfg["optimiser_fee_pct"] = fee_dec
    cfg["ridge_cost_pct"] = fee_dec
    cfg["limit_fill_fee_bps"] = side_bps

    # New fee structure for limit orders
    cfg["fee_bps_market"] = MARKET_ORDER_FEE_BPS
    cfg["fee_bps_limit_entry"] = LIMIT_ORDER_FEE_BPS
    cfg["fee_bps_limit_exit"] = LIMIT_ORDER_FEE_BPS
    cfg["fee_bps_market_exit"] = MARKET_ORDER_FEE_BPS

    # Enable MAE/MFE-based limit offset estimation
    cfg["use_mae_mfe_limit_offset"] = True
    cfg["use_exit_limit_orders"] = True


def _append_suffix(path: str, suffix: str) -> str:
    norm = path.rstrip("/\\")
    if norm.endswith(suffix):
        return norm
    return f"{norm}{suffix}"


def _resolve_path(base_dir: str, path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base_dir, path))


def _normalize_cfg_paths(cfg: dict) -> None:
    """
    Normalize relative config paths to stable absolute paths independent of cwd.
    """
    # Resolve paths relative to the project root (parent of this script's directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cfg["data_root"] = _resolve_path(project_root, str(cfg.get("data_root", "data")))
    cfg["reports_root"] = _resolve_path(
        project_root, str(cfg.get("reports_root", "reports"))
    )
    cfg["hf_data_dir"] = _resolve_path(
        project_root, str(cfg.get("hf_data_dir", "15m_ohlcv"))
    )


def _configure_report_roots(cfg: dict) -> None:
    report_root = cfg.get("reports_root")
    if report_root:
        os.environ["EPM_REPORTS_DIR"] = str(report_root)


def _load_mask_params_by_mode(cfg: dict) -> dict:
    """Refresh cfg with persisted offline optimizer params (including mask params by mode)."""
    merged = apply_offline_optimizer_best_params(dict(cfg))
    cfg.update(merged)
    return dict(cfg.get("candidate_mask_params_by_mode", {}) or {})


def _downcast_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce optimiser memory footprint."""
    for col in df.columns:
        dt = df[col].dtype
        if pd.api.types.is_float_dtype(dt):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(dt):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def _resolve_ts_sig(cfg: dict, ts_override=None) -> pd.Timestamp:
    if ts_override:
        try:
            _ts_str = (
                str(ts_override).split("_v")[0]
                if "_v" in str(ts_override)
                else str(ts_override)
            )
            ts_sig = pd.to_datetime(_ts_str, format="%Y%m%d_%H%M%S").tz_localize("UTC")
        except ValueError:
            ts_sig = pd.Timestamp(ts_override).tz_localize("UTC")
    else:
        ts_sig = _find_latest_feature_ts(cfg.get("data_root", "data"))
    return ts_sig


def _find_latest_feature_ts(data_root):
    """Find the latest feature timestamp directory."""
    import glob
    import os

    feat_dir = os.path.join(data_root, "features")
    if not os.path.exists(feat_dir):
        return None
    dirs = sorted(glob.glob(os.path.join(feat_dir, "20*")))
    if not dirs:
        return None
    latest = os.path.basename(dirs[-1])
    return pd.to_datetime(latest, format="%Y%m%d_%H%M%S").tz_localize("UTC")


def run_download(cfg):
    """Download OHLCV data from Binance for the full training universe."""
    cfg.setdefault("allow_15m_download", False)
    import time as _time

    from extreme_price_movements.hf_data_loader import sync_15m_ohlcv_range

    tprint("STEP: DOWNLOAD START")
    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])
    use_perps = bool(cfg.get("use_perps", False))
    _check_complete = str(
        os.environ.get(
            "EPM_DOWNLOAD_CHECK_COMPLETE", str(cfg.get("download_check_complete", True))
        )
    ).strip().lower() in {"1", "true", "yes", "y", "on"}
    _missing_lt_days = float(
        os.environ.get(
            "EPM_DOWNLOAD_SKIP_LT_DAYS",
            cfg.get("download_skip_if_missing_lt_days", 3.0),
        )
        or 0.0
    )

    # --- Freshness check: skip download if data is < 6 days old ---
    import glob as _glob
    import json as _json

    _FRESHNESS_DAYS = 6
    _force_download = str(
        os.environ.get("EPM_DOWNLOAD_FORCE", str(cfg.get("download_force", False)))
    ).strip().lower() in {"1", "true", "yes", "y", "on"}
    _meta_dir = store.ohlcv_dir
    _meta_files = _glob.glob(os.path.join(_meta_dir, "*.meta.json"))
    if _meta_files and (not _force_download) and (not _check_complete):
        _latest_ms = 0
        for mf in _meta_files[:20]:  # sample up to 20 symbols
            try:
                with open(mf) as _fp:
                    _m = _json.load(_fp)
                _latest_ms = max(_latest_ms, _m.get("last_ts_ms", 0))
            except Exception:
                pass
        if _latest_ms > 0:
            _latest_ts = pd.to_datetime(_latest_ms, unit="ms", utc=True)
            _age = pd.Timestamp.utcnow() - _latest_ts
            tprint(f"Data freshness: latest={_latest_ts}, age={_age}")
            if _age < pd.Timedelta(days=_FRESHNESS_DAYS):
                tprint(
                    f"Data is {_age.total_seconds()/3600:.1f}h old (< {_FRESHNESS_DAYS}d). Skipping download."
                )
                tprint("STEP: DOWNLOAD COMPLETE (fresh)")
                return
    elif _force_download:
        tprint("Freshness gate bypassed (download_force=True)")

    ex = make_perp_exchange() if use_perps else make_spot_exchange()

    mu = refresh_margin_universe_daily(None, quotes=("USDT", "USDC", "BUSD", "EUR"))
    fetch_syms = build_fetch_universe(
        mu.symbols, cfg["market_basket"], cfg["fetch_symbols_M"]
    )
    _base_n = len(fetch_syms)

    # Runtime overrides for parallel download orchestrations.
    _order = (
        str(
            os.environ.get(
                "EPM_DOWNLOAD_SYMBOL_ORDER", cfg.get("download_symbol_order", "volume")
            )
        )
        .strip()
        .lower()
    )
    _stride = max(
        1,
        int(
            os.environ.get(
                "EPM_DOWNLOAD_SYMBOL_STRIDE", cfg.get("download_symbol_stride", 1)
            )
        ),
    )
    _offset = max(
        0,
        int(
            os.environ.get(
                "EPM_DOWNLOAD_SYMBOL_OFFSET", cfg.get("download_symbol_offset", 0)
            )
        ),
    )
    _max_symbols = int(
        os.environ.get("EPM_DOWNLOAD_MAX_SYMBOLS", cfg.get("download_max_symbols", 0))
        or 0
    )
    _part_count = max(
        1,
        int(
            os.environ.get(
                "EPM_DOWNLOAD_PARTITION_COUNT", cfg.get("download_partition_count", 1)
            )
        ),
    )
    _part_id = max(
        0,
        int(
            os.environ.get(
                "EPM_DOWNLOAD_PARTITION_ID", cfg.get("download_partition_id", 0)
            )
        ),
    )
    if _part_id >= _part_count:
        _part_id = _part_count - 1

    # Disjoint partitioning is based on canonical alpha order to avoid overlap
    # between multiple concurrent downloaders.
    _alpha = sorted(fetch_syms)
    if _part_count > 1:
        _selected = [s for i, s in enumerate(_alpha) if (i % _part_count) == _part_id]
    else:
        _selected = _alpha

    if _order in {"alpha_desc", "reverse_alpha", "reverse_alphabetical"}:
        fetch_syms = sorted(_selected, reverse=True)
    elif _order in {"alpha_asc", "alphabetical"}:
        fetch_syms = sorted(_selected)
    else:
        _sel = set(_selected)
        fetch_syms = [s for s in fetch_syms if s in _sel]

    if _stride > 1 or _offset > 0:
        fetch_syms = fetch_syms[_offset::_stride]

    if _max_symbols > 0:
        fetch_syms = fetch_syms[:_max_symbols]

    tprint(
        f"Download universe: {len(fetch_syms)} symbols "
        f"(base={_base_n}, order={_order}, stride={_stride}, offset={_offset}, "
        f"partition={_part_id}/{_part_count}, max={_max_symbols if _max_symbols > 0 else 'all'})"
    )

    fetch_years = cfg.get("fetch_years", 3)
    since = pd.Timestamp.utcnow() - pd.Timedelta(days=int(fetch_years * 365))
    since_ms = int(since.value // 10**6)
    now_utc = pd.Timestamp.now(tz="UTC")
    since_1h = since.floor("1h")
    now_1h = now_utc.floor("1h")
    since_15m = since.floor("15min")
    now_15m = now_utc.floor("15min")

    def _panel_complete(
        df: Optional[pd.DataFrame],
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        freq: str,
    ) -> bool:
        if df is None or df.empty:
            return False
        idx = (
            df.index
            if isinstance(df.index, pd.DatetimeIndex)
            else pd.to_datetime(df.index)
        )
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        window = idx[(idx >= start_ts) & (idx <= end_ts)]
        if len(window) == 0:
            return False
        expected = len(pd.date_range(start=start_ts, end=end_ts, freq=freq, tz="UTC"))
        return (
            (len(window) == expected)
            and (window.min() <= start_ts)
            and (window.max() >= end_ts)
        )

    def _panel_missing_days(
        df: Optional[pd.DataFrame],
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        freq: str,
    ) -> float:
        expected = len(pd.date_range(start=start_ts, end=end_ts, freq=freq, tz="UTC"))
        if expected <= 0:
            return 0.0
        if df is None or df.empty:
            return float((end_ts - start_ts) / pd.Timedelta(days=1))
        idx = (
            df.index
            if isinstance(df.index, pd.DatetimeIndex)
            else pd.to_datetime(df.index)
        )
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        window = idx[(idx >= start_ts) & (idx <= end_ts)]
        observed = len(pd.DatetimeIndex(window).unique())
        missing_bars = max(0, expected - observed)
        step = pd.to_timedelta(freq)
        return float((missing_bars * step) / pd.Timedelta(days=1))

    def _symbol_status_1h(sym: str) -> Tuple[bool, float]:
        try:
            df_local = store.load(sym)
            return (
                _panel_complete(df_local, since_1h, now_1h, "1h"),
                _panel_missing_days(df_local, since_1h, now_1h, "1h"),
            )
        except Exception:
            return False, 1e9

    def _symbol_status_15m(sym: str) -> Tuple[bool, float]:
        try:
            from extreme_price_movements.hf_data_loader import _load_existing_data

            df_local = _load_existing_data(sym)
            return (
                _panel_complete(df_local, since_15m, now_15m, "15min"),
                _panel_missing_days(df_local, since_15m, now_15m, "15min"),
            )
        except Exception:
            return False, 1e9

    success_1h, fail_1h = 0, 0
    success_15m, fail_15m = 0, 0
    skip_1h, skip_15m = 0, 0
    skip_small_1h, skip_small_15m = 0, 0
    for i, sym in enumerate(fetch_syms):
        if _check_complete:
            complete_1h, missing_1h_days = _symbol_status_1h(sym)
            complete_15m, missing_15m_days = _symbol_status_15m(sym)
        else:
            complete_1h, complete_15m = False, False
            missing_1h_days, missing_15m_days = 1e9, 1e9

        if complete_1h:
            skip_1h += 1
        elif missing_1h_days < _missing_lt_days:
            skip_1h += 1
            skip_small_1h += 1
        else:
            try:
                if use_perps:
                    store.update_symbol_perp(ex, sym, since_ms)
                else:
                    store.update_symbol(ex, sym, since_ms)
                success_1h += 1
            except Exception as e:
                fail_1h += 1
                tprint(f"  FAIL 1h {sym}: {e}")

        if complete_15m:
            skip_15m += 1
        elif missing_15m_days < _missing_lt_days:
            skip_15m += 1
            skip_small_15m += 1
        else:
            try:
                df_15m = sync_15m_ohlcv_range(
                    ex,
                    sym,
                    since,
                    pd.Timestamp.now(tz="UTC"),
                    full_backfill=bool(cfg.get("download_15m_full_backfill", True)),
                )
                if df_15m is None or df_15m.empty:
                    fail_15m += 1
                    tprint(f"  FAIL 15m {sym}: empty range")
                else:
                    success_15m += 1
            except Exception as e:
                fail_15m += 1
                tprint(f"  FAIL 15m {sym}: {e}")

        try:
            if (i + 1) % 10 == 0:
                tprint(
                    f"  Download progress: {i+1}/{len(fetch_syms)} "
                    f"(1h ok={success_1h}, 1h skip={skip_1h} [<{_missing_lt_days:g}d={skip_small_1h}], 1h fail={fail_1h}, "
                    f"15m ok={success_15m}, 15m skip={skip_15m} [<{_missing_lt_days:g}d={skip_small_15m}], 15m fail={fail_15m})"
                )
        except Exception:
            pass
        _time.sleep(0.1)  # gentle rate limit

    tprint(
        f"STEP: DOWNLOAD COMPLETE — symbols={len(fetch_syms)} "
        f"(1h ok={success_1h}, 1h skip={skip_1h} [<{_missing_lt_days:g}d={skip_small_1h}], 1h fail={fail_1h}; "
        f"15m ok={success_15m}, 15m skip={skip_15m} [<{_missing_lt_days:g}d={skip_small_15m}], 15m fail={fail_15m})"
    )


def _label_artifacts_ready(cfg, ts_sig):
    """Check whether core label artifacts exist for this run timestamp."""
    import os

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    horizons = cfg.get("label_horizons_hours", [])
    required = [
        "exhaustion_history",
    ]
    from extreme_price_movements.strategy_registry import get_strategies

    strategies = get_strategies(cfg)
    for h in horizons:
        for strat in strategies:
            side = strat["trade_side"]
            k = strat["strategy_id"]
            required.append(f"train_{k}_{h}")

    for name in required:
        fpath = os.path.join(
            cfg["data_root"], "artifacts", run_id, "labels", f"{name}.parquet"
        )
        if not os.path.exists(fpath):
            return False
    return True


def _gc_checkpoint(tag: str) -> int:
    """Trigger GC and emit a short checkpoint log."""
    import gc

    collected = gc.collect()
    tprint(f"GC[{tag}]: collected={collected}")
    return collected


def _cache_checkpoint(tag: str) -> None:
    """Clear known runtime cache directories only if memory is running low."""
    import shutil

    import psutil

    mem = psutil.virtual_memory()
    # Only blast cache if available memory is under 25% or we have less than 4GB free
    if mem.percent < 75.0 and mem.available > 4 * 1024 * 1024 * 1024:
        tprint(
            f"CACHE[{tag}]: skipped cache wipe (mem_avail={mem.available/1e9:.1f}GB)"
        )
        return

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dirs = [
        os.path.join(project_root, "cache"),
        os.path.join(project_root, "data_cache"),
    ]
    for cdir in cache_dirs:
        if os.path.exists(cdir):
            try:
                shutil.rmtree(cdir)
                tprint(f"CACHE[{tag}]: cleared {cdir}")
            except Exception as e:
                tprint(f"CACHE[{tag}]: failed {cdir}: {e}")


def _maintenance_checkpoint(tag: str) -> None:
    """Run cache cleanup + GC checkpoint."""
    _cache_checkpoint(tag)
    _gc_checkpoint(tag)


def run_labels(cfg, horizons=None, ts_override=None, store=None):
    _maintenance_checkpoint("labels:start")
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        tprint("ERROR: No feature directories found. Run feature_generation first.")
        return

    tprint(f"Labels mode. ts_sig={ts_sig} horizons={horizons}")
    _load_mask_params_by_mode(cfg)

    if store is None:
        store = PartitionedOHLCVStore(
            root_dir=cfg["data_root"], timeframe=cfg["timeframe"]
        )

    # No exchange needed — data already in store, features already on disk
    horizons = horizons or [2, 4, 8]
    run_label_generation_step_v2(ts_sig, None, cfg, store, None, horizons=horizons)

    tprint("LABELS PIPELINE COMPLETE")
    _maintenance_checkpoint("labels:end")


def run_features(cfg, ts_override=None, force_recompute: bool = False, store=None):
    _maintenance_checkpoint("features:start")
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        ts_sig = pd.Timestamp.utcnow().floor("h")
    tprint(f"Features mode. Target ts_sig={ts_sig}")
    _load_mask_params_by_mode(cfg)

    if store is None:
        store = PartitionedOHLCVStore(
            root_dir=cfg["data_root"], timeframe=cfg["timeframe"]
        )

    # Pass None for margin_symbols to trigger auto-refresh in universe logic
    run_feature_generation_step(
        ts_sig, None, cfg, store, force_full_recompute=bool(force_recompute)
    )

    tprint("FEATURES PIPELINE COMPLETE")
    _maintenance_checkpoint("features:end")


def run_backtest(cfg, ts_override=None, store=None):
    _maintenance_checkpoint("backtest:start")
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        tprint("ERROR: No feature directories found.")
        return

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    import os

    state_file = os.path.join(
        cfg["data_root"], "artifacts", run_id, "models", "trained_state.pkl"
    )
    if not os.path.exists(state_file):
        tprint(
            f"ERROR: Trained state not found at {state_file}. Run 'train' mode first."
        )
        return

    tprint(f"Backtest mode. ts_sig={ts_sig}")
    if store is None:
        store = PartitionedOHLCVStore(
            root_dir=cfg["data_root"], timeframe=cfg["timeframe"]
        )
    run_backtest_step(ts_sig, None, cfg, store, state_file)
    tprint("BACKTEST PIPELINE COMPLETE")
    _maintenance_checkpoint("backtest:end")


def run_inference_backtest(cfg, ts_override=None, store=None):
    """Run inference-aligned walk-forward backtest on unseen holdout periods."""
    _maintenance_checkpoint("inference_backtest:start")
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        tprint("ERROR: No feature directories found.")
        return

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    import os

    # Check for trained state
    state_file = os.path.join(
        cfg["data_root"], "artifacts", run_id, "models", "trained_state.pkl"
    )
    if not os.path.exists(state_file):
        tprint(
            f"ERROR: Trained state not found at {state_file}. Run 'train' mode first."
        )
        return

    tprint(f"Inference backtest mode. ts_sig={ts_sig}")

    # Load trained state
    import pickle

    with open(state_file, "rb") as f:
        state = pickle.load(f)

    # Extract necessary components
    if store is None:
        store = PartitionedOHLCVStore(
            root_dir=cfg["data_root"], timeframe=cfg["timeframe"]
        )

    # Load panel data
    tprint("Loading panel data...")
    panel, symbols = store.load_panel(
        symbols=state.get("symbols", None),
        start_ts=None,
        end_ts=None,
    )
    if panel is None:
        tprint("ERROR: Failed to load panel data.")
        return

    # Load features
    tprint("Loading features...")
    from extreme_price_movements.data_store import load_features_selected

    feats = load_features_selected(
        root_dir=cfg["data_root"],
        ts_sig=ts_sig,
        symbols=symbols,
    )

    # Load mask params by mode
    tprint("Loading mask params by mode...")
    from extreme_price_movements.offline_optimisers import (
        apply_offline_optimizer_best_params,
    )

    mask_params_by_mode = dict(cfg.get("candidate_mask_params_by_mode", {}) or {})
    if not mask_params_by_mode:
        # Try to load from offline optimizer results
        mask_params = apply_offline_optimizer_best_params(cfg)
        mask_params_by_mode = dict(
            mask_params.get("candidate_mask_params_by_mode", {}) or {}
        )

    # Load strategy exit params
    strategy_exit_params = dict(
        cfg.get("strategy_exit_params", cfg.get("bucket_exit_params", {})) or {}
    )

    # Load trades from state or backtest results
    tprint("Loading trade candidates...")
    trades = state.get("trades")
    if trades is None:
        # Try to load from backtest results
        backtest_file = os.path.join(
            cfg["data_root"], "artifacts", run_id, "backtest_results.csv"
        )
        if os.path.exists(backtest_file):
            import pandas as pd

            trades = pd.read_csv(backtest_file)
        else:
            tprint("ERROR: No trades found in state or backtest_results.csv")
            return

    # Run inference backtest
    tprint("Running inference backtest...")
    from extreme_price_movements.inference_backtest import (
        InferenceBacktestConfig,
        run_inference_backtest,
    )
    from extreme_price_movements.periods_symbols_management import SlicePlannerConfig

    # Configure inference backtest
    ib_config = InferenceBacktestConfig(
        fee_round_trip_pct=cfg.get("round_trip_fee_pct", 0.3),
        top_fracs=tuple(
            cfg.get("inference_backtest_top_fracs", (0.10, 0.20, 0.30, 0.40))
        ),
        annual_days=365,
        sizing_mode=cfg.get("inference_backtest_sizing_mode", "linear"),
        base_position_size=cfg.get("inference_backtest_base_position_size", 1.0),
        default_limit_offset_bps=cfg.get(
            "inference_backtest_default_limit_offset_bps", 0.0
        ),
    )

    # Use SlicePlanner for unseen holdout periods
    planner_cfg = SlicePlannerConfig.fast_defaults()

    results = run_inference_backtest(
        trades=trades,
        panel=panel,
        feats=feats,
        mask_params_by_mode=mask_params_by_mode,
        strategy_exit_params=strategy_exit_params,
        config=ib_config,
        planner_cfg=planner_cfg,
    )

    # Save results
    tprint("Saving inference backtest results...")
    reports_root = cfg.get("reports_root", "reports")
    os.makedirs(reports_root, exist_ok=True)
    output_file = os.path.join(reports_root, f"inference_backtest_{run_id}.json")

    import json

    # Convert numpy types to serializable types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results)
    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    tprint(f"Inference backtest results saved to {output_file}")
    tprint(f"Results: {json.dumps(serializable_results, indent=2)}")
    tprint("INFERENCE BACKTEST PIPELINE COMPLETE")
    _maintenance_checkpoint("inference_backtest:end")


def run_train(cfg, ts_override=None, base_only=False, meta_only=False, store=None):
    _maintenance_checkpoint("train:start")
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        tprint("ERROR: No feature directories found. Run feature_generation first.")
        return

    tprint(f"Train mode. ts_sig={ts_sig} base_only={base_only} meta_only={meta_only}")
    _load_mask_params_by_mode(cfg)

    # TP/SL optimisation happens during label generation (see training.generate_label_datasets).
    # Check if labels already exist before refreshing to avoid unnecessary recomputation.
    if store is None:
        store = PartitionedOHLCVStore(
            root_dir=cfg["data_root"], timeframe=cfg["timeframe"]
        )
    if _label_artifacts_ready(cfg, ts_sig):
        tprint("Label artifacts already exist, skipping label refresh...")
    else:
        if meta_only:
            tprint(
                "ERROR: meta_only requested but labels are missing. Run labels mode first."
            )
            return
        tprint(
            "Refreshing labels to optimise TP:SL widths before model training (optimise_tpsl_ratio)..."
        )
        run_label_generation_step_v2(ts_sig, None, cfg, store, None)

    if not _label_artifacts_ready(cfg, ts_sig):
        tprint(
            "ERROR: Label generation did not produce required artifacts. Aborting training."
        )
        return

    state = run_training_step(
        ts_sig,
        cfg,
        store=store,
        margin_symbols=None,
        base_only=base_only,
        meta_only=meta_only,
    )
    if state:
        tprint("TRAINING PIPELINE COMPLETE")

        # Run breakdown diagnostics after base training
        try:
            run_breakdown_diagnostics_integration(cfg, ts_sig)
        except Exception as e:
            tprint(f"WARNING: breakdown diagnostics failed: {e}")
    else:
        tprint("TRAINING PIPELINE FAILED")
    _maintenance_checkpoint("train:end")


def run_risk_opt(
    cfg, ts_override=None, parsed_ts_sig=None, skip_maintenance=False, store=None
):
    if not skip_maintenance:
        _maintenance_checkpoint("risk_opt:start")

    if parsed_ts_sig:
        ts_sig = parsed_ts_sig
    elif ts_override:
        try:
            _ts_str = (
                str(ts_override).split("_v")[0]
                if "_v" in str(ts_override)
                else str(ts_override)
            )
            ts_sig = pd.to_datetime(_ts_str, format="%Y%m%d_%H%M%S").tz_localize("UTC")
        except ValueError:
            ts_sig = pd.Timestamp(ts_override).tz_localize("UTC")
    else:
        ts_sig = _find_latest_feature_ts(cfg["data_root"])
        if ts_sig is None:
            tprint("ERROR: No feature directories found.")
            return

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    import os

    state_file = os.path.join(
        cfg["data_root"], "artifacts", run_id, "models", "trained_state.pkl"
    )

    tprint(f"Risk Optimization mode. ts_sig={ts_sig}")
    if store is None:
        store = PartitionedOHLCVStore(
            root_dir=cfg["data_root"], timeframe=cfg["timeframe"]
        )
    run_risk_optimization_step(ts_sig, None, cfg, store, state_file)
    tprint("RISK OPTIMIZATION COMPLETE")

    if not skip_maintenance:
        _maintenance_checkpoint("risk_opt:end")


def run_sizer(cfg, ts_override=None, store=None):
    """Run configured sizer backend on meta model OOF predictions."""
    _maintenance_checkpoint("sizer:start")
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        tprint("ERROR: No feature directories found.")
        return

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    import os

    state_file = os.path.join(
        cfg["data_root"], "artifacts", run_id, "models", "trained_state.pkl"
    )
    if not os.path.exists(state_file):
        tprint(
            f"ERROR: Trained state not found at {state_file}. Run 'train' mode first."
        )
        return

    tprint(f"Sizer mode (ridge). ts_sig={ts_sig}")
    _load_mask_params_by_mode(cfg)
    result = run_sizer_step(ts_sig, cfg, state_file)
    if result:
        tprint("SIZER COMPLETE — ridge")

        # Generate OOS backtest metrics immediately after sizer training.
        if bool(cfg.get("sizer_run_oos_backtest", True)):
            try:
                tprint("SIZER: running OOS backtest with updated sizer bundle...")
                if store is None:
                    store = PartitionedOHLCVStore(
                        root_dir=cfg["data_root"], timeframe=cfg["timeframe"]
                    )
                bt_cfg = dict(cfg)
                bt_cfg["sizer_oos_mode"] = True

                # We need to downcast the trades DataFrame to float32 before generating the backtest
                # This is a memory optimization
                trades_path = os.path.join(
                    cfg["data_root"],
                    "artifacts",
                    ts_sig.strftime("%Y%m%d_%H%M%S"),
                    "backtest_results.csv",
                )
                if os.path.exists(trades_path):
                    trades = pd.read_csv(trades_path, low_memory=False)
                    trades = _downcast_numeric_frame(trades)
                    trades.to_csv(trades_path, index=False)

                run_backtest_step(ts_sig, None, bt_cfg, store, state_file)
                tprint("SIZER: OOS backtest complete.")
            except Exception as e:
                tprint(f"WARNING: sizer OOS backtest failed: {e}")

        # Run breakdown diagnostics after ridge sizer
        try:
            run_breakdown_diagnostics_integration(cfg, ts_sig)
        except Exception as e:
            tprint(f"WARNING: breakdown diagnostics failed: {e}")

        _maintenance_checkpoint("sizer:end")
        return True
    else:
        tprint("SIZER: No results (possibly no meta OOF predictions found)")
        _maintenance_checkpoint("sizer:end")
        return False


def run_trigger_discovery(cfg, ts_override=None):
    """Run Trigger Discovery (Phase 2.75) via mask_optimiser."""
    _maintenance_checkpoint("trigger_discovery:start")
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        tprint("ERROR: No feature directories found for Trigger Discovery.")
        return False

    ts_str = ts_sig.strftime("%Y%m%d_%H%M%S")
    tprint(f"Trigger Discovery (Phase 2.75) mode. ts_sig={ts_str}")

    # Construct args for mask_optimiser
    args = argparse.Namespace()
    args.data_root = cfg["data_root"]
    args.ts = ts_str
    args.features = None  # Automatic discovery in mask_optimiser
    args.perps = bool(cfg.get("use_perps", False))
    args.max_symbols = cfg.get("mask_opt_max_symbols")
    args.lookback_years = float(cfg.get("mask_opt_lookback_years", 1.5))
    args.horizons = ",".join(map(str, cfg.get("horizons", [1, 2, 4, 8])))
    args.modes = "long,short"  # Refactored side-based modes
    args.diverse_count = int(cfg.get("mask_opt_diverse_count", 4))

    try:
        mask_opt.run_mask_optimization_4modes(args)
        tprint("TRIGGER DISCOVERY COMPLETE")
        return True
    except Exception as e:
        tprint(f"ERROR: Trigger Discovery failed: {e}")
        return False
    finally:
        _maintenance_checkpoint("trigger_discovery:end")


def run_all(cfg, ts_override=None):
    """Run download -> features -> train (includes labels) -> optimise (learn entry) -> sizer -> optimise (sizing) in order.

    Note: 'train' already refreshes labels internally.
    Note: 'optimise' triggers backtest internally if backtest_results.csv is missing,
          then runs the tpsl_optimiser pipeline (TP/SL calibration, loss limiter,
          profit exit, position sizing, holdout evaluation).
    """
    _maintenance_checkpoint("run_all:start")
    # run_download(cfg)  <- User requested only download if explicitly in download mode
    _maintenance_checkpoint("run_all:after_download")

    # Instantiate store once for use across steps
    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])

    run_features(cfg, ts_override=ts_override, store=store)
    _maintenance_checkpoint("run_all:after_features")

    if bool(cfg.get("enable_trigger_discovery_stage", False)):
        success = run_trigger_discovery(cfg, ts_override=ts_override)
        if not success:
            tprint("ERROR: Trigger Discovery stage failed. Aborting pipeline.")
            return
        # RE-LOAD MASK PARAMS: ensure cfg["strategies"] is populated from new winners!
        _load_mask_params_by_mode(cfg)
        _maintenance_checkpoint("run_all:after_trigger_discovery")

    run_train(cfg, ts_override=ts_override, store=store)
    _maintenance_checkpoint("run_all:after_train")

    # 1. Optimise: learn entry policy (fill model + delta) using default sizing/risk
    #    This ensures ridge_sizer sees the correct trade filter.
    tprint("STEP: OPTIMISE (Phase 1 - Entry Policy)")
    success = run_optimise(cfg, ts_override=ts_override, store=store)
    if not success:
        tprint("ERROR: Phase 1 Optimise failed. Aborting pipeline.")
        return
    _maintenance_checkpoint("run_all:after_optimise_phase1")

    # 2. Sizer: learn meta-model weights using the optimized entry policy
    tprint("STEP: SIZER")
    success = run_sizer(cfg, ts_override=ts_override, store=store)
    if not success:
        tprint("ERROR: Sizer step failed. Aborting pipeline.")
        return
    _maintenance_checkpoint("run_all:after_sizer")

    # 3. Optimise: re-run to allow scalar position sizing (Step 40) to use fresh ridge weights
    tprint("STEP: OPTIMISE (Phase 2 - Sizing with Ridge Weights)")
    success = run_optimise(cfg, ts_override=ts_override, store=store)
    if not success:
        tprint("ERROR: Phase 2 Optimise failed. Aborting pipeline.")
        return
    _maintenance_checkpoint("run_all:after_optimise_phase2")

    # Final Summary
    ts_sig = _resolve_ts_sig(cfg, ts_override)

    if ts_sig:
        run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
        import os

        res_path = os.path.join(
            cfg["data_root"], "artifacts", run_id, "backtest_results.csv"
        )
        if os.path.exists(res_path):
            tprint("\n=== FINAL PIPELINE SUMMARY ===")
            try:
                df = pd.read_csv(res_path)
                count = len(df)

                # Gross vs net summary with explicit distinction.
                gross_total = (
                    float(df["gross_ret"].sum())
                    if "gross_ret" in df.columns
                    else float("nan")
                )
                if "net_ret_equity" in df.columns:
                    net_total = float(df["net_ret_equity"].sum())
                elif "pnl" in df.columns:
                    # Legacy backtest output stores net return under `pnl`.
                    net_total = float(df["pnl"].sum())
                else:
                    net_total = float("nan")

                positive_net_share = (
                    float((df["pnl"] > 0).mean())
                    if (count > 0 and "pnl" in df.columns)
                    else float("nan")
                )
                avg_net_per_trade = (
                    (net_total / count)
                    if count > 0 and pd.notna(net_total)
                    else float("nan")
                )

                if pd.notna(gross_total):
                    tprint(f"Total Gross Return: {gross_total:.4f}")
                tprint(
                    f"Total Net Return: {net_total:.4f}"
                    if pd.notna(net_total)
                    else "Total Net Return: n/a"
                )
                tprint(f"Total Trades: {count}")
                if pd.notna(positive_net_share):
                    tprint(f"Positive-Net Share: {positive_net_share:.2%}")
                if pd.notna(avg_net_per_trade):
                    tprint(f"Avg Net Return per Trade: {avg_net_per_trade:.4f}")
                tprint("==============================\n")
            except Exception as e:
                tprint(f"Could not read results for summary: {e}")
    _maintenance_checkpoint("run_all:end")


def run_train_meta(cfg, ts_override=None, store=None):
    """Re-run only meta model training, reusing existing base models."""
    _maintenance_checkpoint("train_meta:start")
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        tprint("ERROR: No feature directories found.")
        return

    _load_mask_params_by_mode(cfg)
    from extreme_price_movements.main import train_daily_meta

    if store is None:
        store = PartitionedOHLCVStore(
            root_dir=cfg["data_root"], timeframe=cfg["timeframe"]
        )

    # Verify that before training the meta model, we optimise the TP & SL values.
    tprint("Optimising TP:SL before meta-training...")
    run_risk_opt(cfg, parsed_ts_sig=ts_sig, skip_maintenance=True, store=store)

    ex = (
        make_perp_exchange()
        if bool(cfg.get("use_perps", False))
        else make_spot_exchange()
    )
    result = train_daily_meta(ts_sig, None, cfg, store, ex)
    if result:
        import gc

        import joblib

        run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
        models_dir = os.path.join(cfg["data_root"], "artifacts", run_id, "models")
        os.makedirs(models_dir, exist_ok=True)
        meta_state_path = os.path.join(models_dir, "model_state_meta.pkl")

        joblib.dump(result, meta_state_path)
        tprint(f"Meta model state saved to {meta_state_path} using joblib")

        # Free memory before moving on
        del result
        gc.collect()

        # NOTE: Breakdown diagnostics removed here as they require the ridge sizer
        # to represent the final trading policy correctly.
        # Meta-layer metrics (AUC, Lift, IC) are logged naturally during train_daily_meta.

        tprint("TRAIN_META PIPELINE COMPLETE")
    else:
        tprint("TRAIN_META PIPELINE FAILED")
    _maintenance_checkpoint("train_meta:end")


def run_optimise(cfg, ts_override=None, store=None):
    _maintenance_checkpoint("optimise:start")
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    _load_mask_params_by_mode(cfg)
    if ts_sig is None:
        tprint("ERROR: No feature directories found.")
        return False

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    if bool(cfg.get("optimise_use_ridge_oof", False)):
        try:
            run_optimise_from_ridge_oof(
                run_id=run_id,
                data_root=cfg["data_root"],
                fee_roundtrip=float(cfg.get("optimiser_fee_pct", 0.003)),
                cooldown_hours=float(cfg.get("optimise_ridge_oof_cooldown_hours", 0.0)),
            )
        except Exception as exc:
            tprint(f"ERROR: Ridge OOF optimise failed: {exc}")
            return False
        _maintenance_checkpoint("optimise:end")
        return True

    import os

    state_file = os.path.join(
        cfg["data_root"], "artifacts", run_id, "models", "trained_state.pkl"
    )
    backtest_file = os.path.join(
        cfg["data_root"], "artifacts", run_id, "backtest_results.csv"
    )
    if not os.path.exists(backtest_file):
        tprint(
            "Backtest results not found. Running backtest to generate trade data for optimiser..."
        )
        bt_cfg = dict(cfg)
        bt_cfg["offline_backtest_skip_universe_refresh"] = True
        run_backtest(bt_cfg, ts_override=ts_override, store=store)
        if not os.path.exists(backtest_file):
            tprint(
                f"ERROR: Backtest still not found at {backtest_file}. Aborting optimise."
            )
            return False
    trades = pd.read_csv(backtest_file, low_memory=False)
    trades = _downcast_numeric_frame(trades)
    trades.attrs["threaded_exit_stream"] = True  # Inject attribute stripped by CSV save
    if "optimiser_fee_pct" in cfg:
        try:
            trades.attrs["fee_pct"] = float(cfg["optimiser_fee_pct"])
        except Exception:
            pass
    if "atr_pct_15m" in trades.columns:
        atr_15m = trades["atr_pct_15m"]
    elif "atr" in trades.columns:
        atr_15m = trades["atr"]
    else:
        atr_15m = pd.Series(0.01, index=trades.index)

    params_path = os.path.join(
        cfg["data_root"], "artifacts", run_id, "models", "strategy_params.json"
    )
    run_optimise_step(
        trades=trades,
        atr_15m=atr_15m,
        output_path=params_path,
        policy=Policy(mode="train_baseline", params_path=params_path),
        state_path=state_file if os.path.exists(state_file) else None,
        store_base_dir=cfg.get("data_root"),
        run_id=run_id,
        data_root=cfg["data_root"],
        ohlcv_store=store,
    )
    tprint(f"OPTIMISE COMPLETE: {params_path}")

    # Run breakdown diagnostics after optimization
    try:
        run_breakdown_diagnostics_integration(cfg, ts_sig)
    except Exception as e:
        tprint(f"WARNING: breakdown diagnostics failed: {e}")

    try:
        from extreme_price_movements.reports.bucket_report import report_optimise

        run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
        rp = report_optimise(run_id, cfg["data_root"], base_dir=cfg.get("reports_root"))
        tprint(f"Optimise strategy report: {rp}")
    except Exception as _re:
        tprint(f"WARNING: optimise strategy report failed: {_re}")
    _maintenance_checkpoint("optimise:end")
    return True


def clear_caches():
    """Force garbage collection and clear the on-disk caches before a run."""
    import gc
    import os
    import shutil

    # 1. Force Python garbage collection
    collected = gc.collect()
    tprint(f"GC: Collected {collected} objects on startup.")

    # 2. Clear known temporary cache directories
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dirs = [
        os.path.join(project_root, "cache"),
        os.path.join(project_root, "data_cache"),
    ]

    for cdir in cache_dirs:
        if os.path.exists(cdir):
            try:
                shutil.rmtree(cdir)
                tprint(f"CACHE: Cleared directory {cdir}")
            except Exception as e:
                tprint(f"CACHE: Failed to clear {cdir}: {e}")


def run_breakdown_diagnostics_integration(cfg: dict, ts_sig: pd.Timestamp) -> None:
    """Run breakdown diagnostics integrated into pipeline steps."""
    from extreme_price_movements.breakdown_diagnostics import run_breakdown_diagnostics

    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg["data_root"], "artifacts", run_id)

    # Check if OHLC data exists for diagnostics
    ohlc_path = os.path.join(run_dir, "ohlc.parquet")
    if not os.path.exists(ohlc_path):
        # Try to create OHLC from store if missing
        try:
            store = PartitionedOHLCVStore(
                root_dir=cfg["data_root"], timeframe=cfg["timeframe"]
            )
            # Get a representative symbol for OHLC extraction (store has no list_symbols API).
            symbols = []
            ohlcv_dir = getattr(store, "ohlcv_dir", None)
            if ohlcv_dir and os.path.isdir(ohlcv_dir):
                import glob

                for path in glob.glob(os.path.join(ohlcv_dir, "symbol=*")):
                    base = os.path.basename(path)
                    if not base.startswith("symbol="):
                        continue
                    raw = base.replace("symbol=", "")
                    symbols.append(raw.replace("_", "/", 1))
            symbols = sorted(set(symbols))

            if symbols:
                symbol = symbols[0]  # Use first available symbol
                ohlc_data = store.load(symbol)
                if ohlc_data is not None and len(ohlc_data) > 0:
                    ohlc_data.to_parquet(ohlc_path)
                    tprint(f"Created OHLC data for diagnostics from {symbol}")
                else:
                    tprint("WARNING: No OHLC data available for breakdown diagnostics")
                    return
            else:
                tprint("WARNING: No symbols found in store for breakdown diagnostics")
                return
        except Exception as e:
            tprint(f"WARNING: Could not create OHLC data for diagnostics: {e}")
            return

    # Configure breakdown diagnostics
    diag_cfg = {
        "ohlc_path": ohlc_path,
        "lookback_h": cfg.get("breakdown_lookback_h", 12),
        "baseline_trigger": cfg.get("breakdown_trigger", 0.08),
        "trigger_sweep": cfg.get(
            "breakdown_trigger_sweep", [0.06, 0.07, 0.08, 0.09, 0.10]
        ),
        "decluster_h": cfg.get("breakdown_decluster_h", 6),
        "max_event_h": cfg.get("breakdown_max_event_h", 72),
        "entry_offsets_h": cfg.get(
            "breakdown_entry_offsets", [-12, -6, -4, -2, -1, 0, 1, 2, 4, 6, 12]
        ),
        "directions": cfg.get("breakdown_directions", ["follow", "fade"]),
        "cost_stress_multipliers": cfg.get(
            "breakdown_cost_stress", [1.0, 1.25, 1.5, 2.0]
        ),
        "optimise_run_dir": run_dir,
    }

    try:
        tprint("Running breakdown diagnostics...")
        result = run_breakdown_diagnostics(diag_cfg, run_dir)

        # Log key verdicts
        verdict = result.get("verdict", {})
        tprint("BREAKDOWN DIAGNOSTICS VERDICT:")
        for key, value in verdict.items():
            if key == "recommendations":
                continue
            tprint(f"  {key}: {value}")

        recommendations = verdict.get("recommendations", {})
        if recommendations:
            tprint("RECOMMENDATIONS:")
            for key, rec in recommendations.items():
                if verdict.get(
                    key, False
                ):  # Only show recommendations for failed checks
                    tprint(f"  {key}: {rec}")

        tprint(f"Breakdown diagnostics saved to: {run_dir}/breakdown_diagnostics/")

    except Exception as e:
        tprint(f"ERROR: breakdown diagnostics failed: {e}")
        raise


def run_breakdown_diagnostics_standalone(cfg: dict, ts_override: str = None) -> None:
    """Standalone breakdown diagnostics mode."""
    ts_sig = _resolve_ts_sig(cfg, ts_override)
    if ts_sig is None:
        tprint("ERROR: No feature directories found.")
        return

    tprint(f"Breakdown Diagnostics mode. ts_sig={ts_sig}")
    run_breakdown_diagnostics_integration(cfg, ts_sig)
    tprint("BREAKDOWN DIAGNOSTICS COMPLETE")


def main():
    clear_caches()
    parser = argparse.ArgumentParser(description="Extreme Price Movements Pipeline")
    parser.add_argument(
        "mode",
        choices=[
            "download",
            "labels",
            "features",
            "train",
            "train_base",
            "train_meta",
            "sizer",
            "optimise",
            "backtest",
            "inference_backtest",
            "run",
            "breakdown_diagnostics",
        ],
        help="Pipeline mode to run",
    )
    parser.add_argument(
        "-perps",
        "--perps",
        action="store_true",
        help="Run pipeline in perps mode (isolated *_perp roots)",
    )
    parser.add_argument(
        "--force-feature-recompute",
        action="store_true",
        help="Force full recompute in features mode",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Horizons to use for labels/training",
    )
    parser.add_argument(
        "--ts", dest="ts_override", help="Timestamp override (YYYYMMDD_HHMMSS)"
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Only train base models (alpha, spike, exh)",
    )
    parser.add_argument(
        "--meta-only",
        action="store_true",
        help="Only train meta models (runs train_meta)",
    )
    parser.add_argument(
        "--robust-mode",
        action="store_true",
        help="Use robust planner mode (enables full inference retrain)",
    )
    parser.add_argument(
        "--enable-trigger-discovery-stage",
        action="store_true",
        help="Enable the trigger discovery stage in the pipeline",
    )
    parser.add_argument(
        "--optimise-use-ridge-oof",
        action="store_true",
        help="Run optimise in cheap Ridge/limit-offset OOF mode instead of using backtest_results.csv",
    )
    args = parser.parse_args()

    cfg = CFG.copy()
    _apply_fee_model(cfg, BASE_ROUND_TRIP_FEE_PCT)
    _normalize_cfg_paths(cfg)
    if args.perps:
        cfg["use_perps"] = True
        cfg["data_root"] = _append_suffix(cfg.get("data_root", "data"), "_perp")
        cfg["reports_root"] = _append_suffix(
            cfg.get("reports_root", "reports"), "_perp"
        )
        cfg["hf_data_dir"] = _append_suffix(
            cfg.get("hf_data_dir", "15m_ohlcv"), "_perp"
        )
        os.environ["EPM_HF_DATA_DIR"] = str(cfg["hf_data_dir"])
        cfg = enable_perp_feature_keys(cfg)
        # Perp-mode fee model: 0.10% round-trip (5 bps/side).
        _apply_fee_model(cfg, PERP_ROUND_TRIP_FEE_PCT)

    _configure_report_roots(cfg)
    cfg["optimise_use_ridge_oof"] = bool(args.optimise_use_ridge_oof)
    cfg["slice_planner_preset"] = "robust" if bool(args.robust_mode) else "fast"
    cfg["train_full_inference_models"] = bool(args.robust_mode)

    tprint(
        f"Planner preset: {cfg['slice_planner_preset']} (full_inference_retrain={cfg['train_full_inference_models']})"
    )
    cfg["enable_trigger_discovery_stage"] = bool(args.enable_trigger_discovery_stage)

    if args.mode == "download":
        run_download(cfg)
    elif args.mode == "labels":
        run_labels(cfg, horizons=args.horizons, ts_override=args.ts_override)
    elif args.mode == "features":
        run_features(
            cfg,
            ts_override=args.ts_override,
            force_recompute=bool(args.force_feature_recompute),
        )
    elif args.mode == "train":
        run_train(
            cfg,
            ts_override=args.ts_override,
            base_only=args.base_only,
            meta_only=args.meta_only,
        )
    elif args.mode == "train_base":
        run_train(cfg, ts_override=args.ts_override, base_only=True, meta_only=False)
    elif args.mode == "train_meta":
        run_train_meta(cfg, ts_override=args.ts_override)
    elif args.mode == "sizer":
        run_sizer(cfg, ts_override=args.ts_override)
    elif args.mode == "optimise":
        run_optimise(cfg, ts_override=args.ts_override)
    elif args.mode == "backtest":
        run_backtest(cfg, ts_override=args.ts_override)
    elif args.mode == "inference_backtest":
        run_inference_backtest(cfg, ts_override=args.ts_override)
    elif args.mode == "breakdown_diagnostics":
        run_breakdown_diagnostics_standalone(cfg, ts_override=args.ts_override)
    elif args.mode == "run":
        run_all(cfg, ts_override=args.ts_override)


if __name__ == "__main__":
    main()
