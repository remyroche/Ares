#!/usr/bin/env python3
"""
CLI entry point for extreme_price_movements pipeline.

Usage:
    python3 extreme_price_movements/run_pipeline.py labels
"""
import sys
import os

# Avoid expensive/warning-prone Matplotlib cache initialization under read-only HOME.
_mpl_cfg = os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_epm")
try:
    os.makedirs(_mpl_cfg, exist_ok=True)
except Exception:
    pass

# Add parent directory to Python path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import argparse
import pandas as pd

from extreme_price_movements.config import CFG, enable_perp_feature_keys
from extreme_price_movements.utils import tprint
from extreme_price_movements.data_store import (
    PartitionedOHLCVStore,
    make_spot_exchange,
    make_perp_exchange,
)
from extreme_price_movements.universe import refresh_margin_universe_daily, build_fetch_universe
from extreme_price_movements.pipeline_steps import (
    run_label_generation_step_v2,
    run_feature_generation_step,
    run_training_step,
    run_backtest_step,
    run_risk_optimization_step,
)
from extreme_price_movements.optimise import run_optimise_step, Policy
from extreme_price_movements.pipeline_steps import run_sizer_step

# SINGLE SOURCE OF TRUTH FOR FEES - All fee configuration comes from these constants
# Spot trading fees (default)
BASE_ROUND_TRIP_FEE_PCT = 0.3  # 0.3% round-trip = 0.15% per side (15 bps)
# Perpetual trading fees (when --perps flag used)  
PERP_ROUND_TRIP_FEE_PCT = 0.1  # 0.1% round-trip = 0.05% per side (5 bps)



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
    cfg["reports_root"] = _resolve_path(project_root, str(cfg.get("reports_root", "reports")))
    cfg["hf_data_dir"] = _resolve_path(project_root, str(cfg.get("hf_data_dir", "15m_ohlcv")))


def _configure_report_roots(cfg: dict) -> None:
    report_root = cfg.get("reports_root")
    if report_root:
        os.environ["EPM_REPORTS_DIR"] = str(report_root)


def _find_latest_feature_ts(data_root):
    """Find the latest feature timestamp directory."""
    import os, glob
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
    import time as _time
    tprint("STEP: DOWNLOAD START")
    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])
    use_perps = bool(cfg.get("use_perps", False))

    # --- Freshness check: skip download if data is < 6 days old ---
    import glob as _glob, json as _json
    _FRESHNESS_DAYS = 6
    _meta_dir = store.ohlcv_dir
    _meta_files = _glob.glob(os.path.join(_meta_dir, "*.meta.json"))
    if _meta_files:
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
                tprint(f"Data is {_age.total_seconds()/3600:.1f}h old (< {_FRESHNESS_DAYS}d). Skipping download.")
                tprint("STEP: DOWNLOAD COMPLETE (fresh)")
                return

    ex = make_perp_exchange() if use_perps else make_spot_exchange()

    mu = refresh_margin_universe_daily(None, quotes=("USDT", "USDC", "BUSD", "EUR"))
    fetch_syms = build_fetch_universe(mu.symbols, cfg["market_basket"], cfg["fetch_symbols_M"])
    tprint(f"Download universe: {len(fetch_syms)} symbols")

    fetch_years = cfg.get("fetch_years", 3)
    since = pd.Timestamp.utcnow() - pd.Timedelta(days=int(fetch_years * 365))
    since_ms = int(since.value // 10**6)

    success, fail = 0, 0
    for i, sym in enumerate(fetch_syms):
        try:
            if use_perps:
                store.update_symbol_perp(ex, sym, since_ms)
            else:
                store.update_symbol(ex, sym, since_ms)
            success += 1
            if (i + 1) % 10 == 0:
                tprint(f"  Download progress: {i+1}/{len(fetch_syms)} (ok={success}, fail={fail})")
        except Exception as e:
            fail += 1
            tprint(f"  FAIL {sym}: {e}")
        _time.sleep(0.1)  # gentle rate limit

    tprint(f"STEP: DOWNLOAD COMPLETE — {success} ok, {fail} failed out of {len(fetch_syms)}")


def _label_artifacts_ready(cfg, ts_sig):
    """Check whether core label artifacts exist for this run timestamp."""
    import os
    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    horizons = cfg.get("label_horizons_hours", [])
    required = [
        "exhaustion_history",
    ]
    for h in horizons:
        required.extend([
            f"train_long_mr_{h}",
            f"train_long_tf_{h}",
            f"train_short_mr_{h}",
            f"train_short_tf_{h}",
        ])

    for name in required:
        fpath = os.path.join(cfg["data_root"], "artifacts", run_id, "labels", f"{name}.parquet")
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
    """Clear known runtime cache directories and emit a short checkpoint log."""
    import shutil

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

def run_labels(cfg, horizons=None, ts_override=None):
    _maintenance_checkpoint("labels:start")
    if ts_override:
        try:
            ts_sig = pd.to_datetime(ts_override, format="%Y%m%d_%H%M%S").tz_localize("UTC")
        except ValueError:
            ts_sig = pd.Timestamp(ts_override).tz_localize("UTC")
    else:
        ts_sig = _find_latest_feature_ts(cfg["data_root"])
        if ts_sig is None:
            tprint("ERROR: No feature directories found. Run feature_generation first.")
            return

    tprint(f"Labels mode. ts_sig={ts_sig} horizons={horizons}")

    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])

    # No exchange needed — data already in store, features already on disk
    horizons = horizons or [2, 4, 8]
    run_label_generation_step_v2(ts_sig, None, cfg, store, None, horizons=horizons)

    tprint("LABELS PIPELINE COMPLETE")
    _maintenance_checkpoint("labels:end")


def run_features(cfg, ts_override=None, force_recompute: bool = False):
    _maintenance_checkpoint("features:start")
    if ts_override:
        try:
            _ts_str = str(ts_override).split("_v")[0] if "_v" in str(ts_override) else str(ts_override)
            ts_sig = pd.to_datetime(_ts_str, format="%Y%m%d_%H%M%S").tz_localize("UTC")
        except ValueError:
            ts_sig = pd.Timestamp(ts_override).tz_localize("UTC")
    else:
        # Re-use latest existing feature timestamp if available, else current hour
        ts_sig = _find_latest_feature_ts(cfg["data_root"])
        if ts_sig is None:
            ts_sig = pd.Timestamp.utcnow().floor("h")
    tprint(f"Features mode. Target ts_sig={ts_sig}")

    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])

    # Pass None for margin_symbols to trigger auto-refresh in universe logic
    run_feature_generation_step(ts_sig, None, cfg, store, force_full_recompute=bool(force_recompute))

    tprint("FEATURES PIPELINE COMPLETE")
    _maintenance_checkpoint("features:end")


def run_backtest(cfg, ts_override=None):
    _maintenance_checkpoint("backtest:start")
    if ts_override:
        try:
            _ts_str = str(ts_override).split("_v")[0] if "_v" in str(ts_override) else str(ts_override)
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
    state_file = os.path.join(cfg["data_root"], "artifacts", run_id, "models", "trained_state.pkl")
    if not os.path.exists(state_file):
        tprint(f"ERROR: Trained state not found at {state_file}. Run 'train' mode first.")
        return

    tprint(f"Backtest mode. ts_sig={ts_sig}")
    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])
    run_backtest_step(ts_sig, None, cfg, store, state_file)
    tprint("BACKTEST PIPELINE COMPLETE")
    _maintenance_checkpoint("backtest:end")


def run_train(cfg, ts_override=None, base_only=False, meta_only=False):
    _maintenance_checkpoint("train:start")
    if ts_override:
        try:
            _ts_str = str(ts_override).split("_v")[0] if "_v" in str(ts_override) else str(ts_override)
            ts_sig = pd.to_datetime(_ts_str, format="%Y%m%d_%H%M%S").tz_localize("UTC")
        except ValueError:
            ts_sig = pd.Timestamp(ts_override).tz_localize("UTC")
    else:
        ts_sig = _find_latest_feature_ts(cfg["data_root"])
        if ts_sig is None:
            tprint("ERROR: No feature directories found. Run feature_generation first.")
            return

    tprint(f"Train mode. ts_sig={ts_sig} base_only={base_only} meta_only={meta_only}")

    # TP/SL optimisation happens during label generation (see training.generate_label_datasets).
    # Check if labels already exist before refreshing to avoid unnecessary recomputation.
    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])
    if _label_artifacts_ready(cfg, ts_sig):
        tprint("Label artifacts already exist, skipping label refresh...")
    else:
        if meta_only:
             tprint("ERROR: meta_only requested but labels are missing. Run labels mode first.")
             return
        tprint("Refreshing labels to optimise TP:SL widths before model training (optimise_tpsl_ratio)...")
        run_label_generation_step_v2(ts_sig, None, cfg, store, None)

    if not _label_artifacts_ready(cfg, ts_sig):
        tprint("ERROR: Label generation did not produce required artifacts. Aborting training.")
        return

    if meta_only:
        run_train_meta(cfg, ts_override=ts_override)
        return

    state = run_training_step(ts_sig, cfg, store=store, margin_symbols=None, base_only=base_only)
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


def run_risk_opt(cfg, ts_override=None):
    _maintenance_checkpoint("risk_opt:start")
    if ts_override:
        try:
            _ts_str = str(ts_override).split("_v")[0] if "_v" in str(ts_override) else str(ts_override)
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
    state_file = os.path.join(cfg["data_root"], "artifacts", run_id, "models", "trained_state.pkl")

    tprint(f"Risk Optimization mode. ts_sig={ts_sig}")
    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])
    run_risk_optimization_step(ts_sig, None, cfg, store, state_file)
    tprint("RISK OPTIMIZATION COMPLETE")
    _maintenance_checkpoint("risk_opt:end")





def run_sizer(cfg, ts_override=None):
    """Run configured sizer backend on meta model OOF predictions."""
    _maintenance_checkpoint("sizer:start")
    if ts_override:
        try:
            _ts_str = str(ts_override).split("_v")[0] if "_v" in str(ts_override) else str(ts_override)
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
    state_file = os.path.join(cfg["data_root"], "artifacts", run_id, "models", "trained_state.pkl")
    if not os.path.exists(state_file):
        tprint(f"ERROR: Trained state not found at {state_file}. Run 'train' mode first.")
        return

    tprint(f"Sizer mode. ts_sig={ts_sig} backend={cfg.get('position_sizer_backend', 'ridge')}")
    result = run_sizer_step(ts_sig, cfg, state_file)
    if result:
        tprint(f"SIZER COMPLETE — backend={cfg.get('position_sizer_backend', 'ridge')}")
        
        # Run breakdown diagnostics after ridge sizer
        try:
            run_breakdown_diagnostics_integration(cfg, ts_sig)
        except Exception as e:
            tprint(f"WARNING: breakdown diagnostics failed: {e}")
    else:
        tprint("SIZER: No results (possibly no meta OOF predictions found)")
    _maintenance_checkpoint("sizer:end")



def run_ridge_sizer(cfg, ts_override=None):
    """Backward-compatible alias for sizer mode."""
    return run_sizer(cfg, ts_override=ts_override)

def run_all(cfg, ts_override=None):
    """Run download -> features -> train (includes labels) -> optimise (learn entry) -> ridge_sizer -> optimise (sizing) in order.
    
    Note: 'train' already refreshes labels internally.
    Note: 'optimise' triggers backtest internally if backtest_results.csv is missing,
          then runs the tpsl_optimiser pipeline (TP/SL calibration, loss limiter,
          profit exit, position sizing, holdout evaluation).
    """
    _maintenance_checkpoint("run_all:start")
    run_download(cfg)
    _maintenance_checkpoint("run_all:after_download")
    run_features(cfg, ts_override=ts_override)
    _maintenance_checkpoint("run_all:after_features")
    run_train(cfg, ts_override=ts_override)
    _maintenance_checkpoint("run_all:after_train")

    # 1. Optimise: learn entry policy (fill model + delta) using default sizing/risk
    #    This ensures ridge_sizer sees the correct trade filter.
    tprint("STEP: OPTIMISE (Phase 1 - Entry Policy)")
    run_optimise(cfg, ts_override=ts_override)
    _maintenance_checkpoint("run_all:after_optimise_phase1")

    # 2. Ridge Sizer: learn meta-model weights using the optimized entry policy
    tprint("STEP: RIDGE SIZER")
    run_sizer(cfg, ts_override=ts_override)
    _maintenance_checkpoint("run_all:after_sizer")

    # 3. Optimise: re-run to allow scalar position sizing (Step 40) to use fresh ridge weights
    tprint("STEP: OPTIMISE (Phase 2 - Sizing with Ridge Weights)")
    run_optimise(cfg, ts_override=ts_override)
    _maintenance_checkpoint("run_all:after_optimise_phase2")

    # Final Summary
    if ts_override:
        try:
            _ts_str = str(ts_override).split("_v")[0] if "_v" in str(ts_override) else str(ts_override)
            ts_sig = pd.to_datetime(_ts_str, format="%Y%m%d_%H%M%S").tz_localize("UTC")
        except ValueError:
            ts_sig = pd.Timestamp(ts_override).tz_localize("UTC")
    else:
        ts_sig = _find_latest_feature_ts(cfg["data_root"])

    if ts_sig:
        run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
        import os
        res_path = os.path.join(cfg["data_root"], "artifacts", run_id, "backtest_results.csv")
        if os.path.exists(res_path):
            tprint("\n=== FINAL PIPELINE SUMMARY ===")
            try:
                df = pd.read_csv(res_path)
                count = len(df)

                # Gross vs net summary with explicit distinction.
                gross_total = float(df["gross_ret"].sum()) if "gross_ret" in df.columns else float("nan")
                if "net_ret_equity" in df.columns:
                    net_total = float(df["net_ret_equity"].sum())
                elif "pnl" in df.columns:
                    # Legacy backtest output stores net return under `pnl`.
                    net_total = float(df["pnl"].sum())
                else:
                    net_total = float("nan")

                positive_net_share = float((df["pnl"] > 0).mean()) if (count > 0 and "pnl" in df.columns) else float("nan")
                avg_net_per_trade = (net_total / count) if count > 0 and pd.notna(net_total) else float("nan")

                if pd.notna(gross_total):
                    tprint(f"Total Gross Return: {gross_total:.4f}")
                tprint(f"Total Net Return: {net_total:.4f}" if pd.notna(net_total) else "Total Net Return: n/a")
                tprint(f"Total Trades: {count}")
                if pd.notna(positive_net_share):
                    tprint(f"Positive-Net Share: {positive_net_share:.2%}")
                if pd.notna(avg_net_per_trade):
                    tprint(f"Avg Net Return per Trade: {avg_net_per_trade:.4f}")
                tprint("==============================\n")
            except Exception as e:
                tprint(f"Could not read results for summary: {e}")
    _maintenance_checkpoint("run_all:end")


def run_train_meta(cfg, ts_override=None):
    """Re-run only meta model training, reusing existing base models."""
    _maintenance_checkpoint("train_meta:start")
    if ts_override:
        try:
            _ts_str = str(ts_override).split("_v")[0] if "_v" in str(ts_override) else str(ts_override)
            ts_sig = pd.to_datetime(_ts_str, format="%Y%m%d_%H%M%S").tz_localize("UTC")
        except ValueError:
            ts_sig = pd.Timestamp(ts_override).tz_localize("UTC")
    else:
        ts_sig = _find_latest_feature_ts(cfg["data_root"])
        if ts_sig is None:
            tprint("ERROR: No feature directories found.")
            return

    from extreme_price_movements.main import train_daily_meta
    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])
    
    # Verify that before training the meta model, we optimise the TP & SL values.
    tprint("Optimising TP:SL before meta-training...")
    run_risk_opt(cfg, ts_override=ts_override)
    
    ex = make_perp_exchange() if bool(cfg.get("use_perps", False)) else make_spot_exchange()
    result = train_daily_meta(ts_sig, None, cfg, store, ex)
    if result:
        import pickle as _pkl
        run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
        models_dir = os.path.join(cfg["data_root"], "artifacts", run_id, "models")
        os.makedirs(models_dir, exist_ok=True)
        meta_state_path = os.path.join(models_dir, "model_state_meta.pkl")
        with open(meta_state_path, "wb") as f:
            _pkl.dump(result, f)
        tprint(f"Meta model state saved to {meta_state_path}")

        # Backward compatibility for existing spot-only workflows that expect
        # the legacy CWD artifact.
        if not bool(cfg.get("use_perps", False)):
            with open("model_state.pkl", "wb") as f:
                _pkl.dump(result, f)
            tprint("Meta model state also saved to legacy model_state.pkl")
        
        # Run breakdown diagnostics after meta training
        try:
            run_breakdown_diagnostics_integration(cfg, ts_sig)
        except Exception as e:
            tprint(f"WARNING: breakdown diagnostics failed: {e}")
            
        tprint("TRAIN_META PIPELINE COMPLETE")
    else:
        tprint("TRAIN_META PIPELINE FAILED")
    _maintenance_checkpoint("train_meta:end")


def run_optimise(cfg, ts_override=None):
    _maintenance_checkpoint("optimise:start")
    if ts_override:
        try:
            _ts_str = str(ts_override).split("_v")[0] if "_v" in str(ts_override) else str(ts_override)
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
    state_file = os.path.join(cfg["data_root"], "artifacts", run_id, "models", "trained_state.pkl")
    backtest_file = os.path.join(cfg["data_root"], "artifacts", run_id, "backtest_results.csv")
    if not os.path.exists(backtest_file):
        tprint("Backtest results not found. Running backtest to generate trade data for optimiser...")
        run_backtest(cfg, ts_override=ts_override)
        if not os.path.exists(backtest_file):
            tprint(f"ERROR: Backtest still not found at {backtest_file}. Aborting optimise.")
            return
    trades = pd.read_csv(backtest_file)
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

    params_path = os.path.join(cfg["data_root"], "artifacts", run_id, "models", "bucket_params.json")
    run_optimise_step(
        trades=trades, atr_15m=atr_15m, output_path=params_path,
        policy=Policy(mode="train_baseline", params_path=params_path),
        state_path=state_file if os.path.exists(state_file) else None,
        store_base_dir=cfg.get("data_root"),
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
        rp = report_optimise(run_id, cfg["data_root"], base_dir=cfg.get('reports_root'))
        tprint(f"Optimise bucket report: {rp}")
    except Exception as _re:
        tprint(f"WARNING: optimise bucket report failed: {_re}")
    _maintenance_checkpoint("optimise:end")

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
        os.path.join(project_root, "data_cache")
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
            store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])
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
        "trigger_sweep": cfg.get("breakdown_trigger_sweep", [0.06, 0.07, 0.08, 0.09, 0.10]),
        "decluster_h": cfg.get("breakdown_decluster_h", 6),
        "max_event_h": cfg.get("breakdown_max_event_h", 72),
        "entry_offsets_h": cfg.get("breakdown_entry_offsets", [-12, -6, -4, -2, -1, 0, 1, 2, 4, 6, 12]),
        "directions": cfg.get("breakdown_directions", ["follow", "fade"]),
        "cost_stress_multipliers": cfg.get("breakdown_cost_stress", [1.0, 1.25, 1.5, 2.0]),
        "optimise_run_dir": run_dir
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
                if verdict.get(key, False):  # Only show recommendations for failed checks
                    tprint(f"  {key}: {rec}")
        
        tprint(f"Breakdown diagnostics saved to: {run_dir}/breakdown_diagnostics/")
        
    except Exception as e:
        tprint(f"ERROR: breakdown diagnostics failed: {e}")
        raise


def run_breakdown_diagnostics_standalone(cfg: dict, ts_override: str = None) -> None:
    """Standalone breakdown diagnostics mode."""
    if ts_override:
        try:
            ts_sig = pd.to_datetime(ts_override, format="%Y%m%d_%H%M%S").tz_localize("UTC")
        except ValueError:
            ts_sig = pd.Timestamp(ts_override).tz_localize("UTC")
    else:
        ts_sig = _find_latest_feature_ts(cfg["data_root"])
        if ts_sig is None:
            tprint("ERROR: No feature directories found.")
            return
    
    tprint(f"Breakdown Diagnostics mode. ts_sig={ts_sig}")
    run_breakdown_diagnostics_integration(cfg, ts_sig)
    tprint("BREAKDOWN DIAGNOSTICS COMPLETE")

def main():
    clear_caches()
    parser = argparse.ArgumentParser(description="Extreme Price Movements Pipeline")
    parser.add_argument("mode", choices=["download", "labels", "features", "train", "train_meta", "ridge_sizer", "sizer", "backtest", "optimize_risk", "optimise", "run", "breakdown_diagnostics"],
                        help="Pipeline mode to run")
    parser.add_argument("-perps", "--perps", action="store_true", help="Run pipeline in perps mode (isolated *_perp roots)")
    parser.add_argument("--force-feature-recompute", action="store_true", help="Force full recompute in features mode")
    parser.add_argument("--horizons", type=int, nargs="+", default=[2, 4, 8], help="Horizons to use for labels/training")
    parser.add_argument("--ts", dest="ts_override", help="Timestamp override (YYYYMMDD_HHMMSS)")
    parser.add_argument("--base-only", action="store_true", help="Only train base models (alpha, spike, exh)")
    parser.add_argument("--meta-only", action="store_true", help="Only train meta models (runs train_meta)")
    args = parser.parse_args()

    cfg = CFG.copy()
    _apply_fee_model(cfg, BASE_ROUND_TRIP_FEE_PCT)
    _normalize_cfg_paths(cfg)
    if args.perps:
        cfg["use_perps"] = True
        cfg["data_root"] = _append_suffix(cfg.get("data_root", "data"), "_perp")
        cfg["reports_root"] = _append_suffix(cfg.get("reports_root", "reports"), "_perp")
        cfg["hf_data_dir"] = _append_suffix(cfg.get("hf_data_dir", "15m_ohlcv"), "_perp")
        os.environ["EPM_HF_DATA_DIR"] = str(cfg["hf_data_dir"])
        cfg = enable_perp_feature_keys(cfg)
        # Perp-mode fee model: 0.10% round-trip (5 bps/side).
        _apply_fee_model(cfg, PERP_ROUND_TRIP_FEE_PCT)

    _configure_report_roots(cfg)

    tprint(f"Sizer backend resolved: {cfg.get('position_sizer_backend', 'ridge')} (enabled={bool(cfg.get('position_sizer_enabled', False))})")

    if args.mode == "download":
        run_download(cfg)
    elif args.mode == "labels":
        run_labels(cfg, horizons=args.horizons, ts_override=args.ts_override)
    elif args.mode == "features":
        run_features(cfg, ts_override=args.ts_override, force_recompute=bool(args.force_feature_recompute))
    elif args.mode == "train":
        run_train(cfg, ts_override=args.ts_override, base_only=args.base_only, meta_only=args.meta_only)
    elif args.mode == "train_meta":
        run_train_meta(cfg, ts_override=args.ts_override)
    elif args.mode == "ridge_sizer":
        run_ridge_sizer(cfg, ts_override=args.ts_override)
    elif args.mode == "sizer":
        run_sizer(cfg, ts_override=args.ts_override)
    elif args.mode == "backtest":
        run_backtest(cfg, ts_override=args.ts_override)
    elif args.mode == "optimize_risk":
        run_risk_opt(cfg, ts_override=args.ts_override)
    elif args.mode == "optimise":
        run_optimise(cfg, ts_override=args.ts_override)
    elif args.mode == "breakdown_diagnostics":
        run_breakdown_diagnostics_standalone(cfg, ts_override=args.ts_override)
    elif args.mode == "run":
        run_all(cfg, ts_override=args.ts_override)


if __name__ == "__main__":
    main()
