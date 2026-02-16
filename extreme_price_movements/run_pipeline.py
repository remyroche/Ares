#!/usr/bin/env python3
"""
CLI entry point for extreme_price_movements pipeline.

Usage:
    python3 extreme_price_movements/run_pipeline.py labels
"""
import sys
import os
import argparse
import pandas as pd

from extreme_price_movements.config import CFG
from extreme_price_movements.utils import tprint
from extreme_price_movements.data_store import PartitionedOHLCVStore, make_spot_exchange
from extreme_price_movements.universe import refresh_margin_universe_daily, build_fetch_universe
from extreme_price_movements.pipeline_steps import (
    run_label_generation_step_v2,
    run_feature_generation_step,
    run_training_step,
    run_backtest_step,
    run_risk_optimization_step,
)
from extreme_price_movements.optimise import run_optimise_step, Policy
from extreme_price_movements.pipeline_steps import run_ridge_sizer_step


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

    ex = make_spot_exchange()

    mu = refresh_margin_universe_daily(None, quotes=("USDT", "USDC", "BUSD", "EUR"))
    fetch_syms = build_fetch_universe(mu.symbols, cfg["market_basket"], cfg["fetch_symbols_M"])
    tprint(f"Download universe: {len(fetch_syms)} symbols")

    fetch_years = cfg.get("fetch_years", 3)
    since = pd.Timestamp.utcnow() - pd.Timedelta(days=int(fetch_years * 365))
    since_ms = int(since.value // 10**6)

    success, fail = 0, 0
    for i, sym in enumerate(fetch_syms):
        try:
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

def run_labels(cfg, ts_override=None):
    if ts_override:
        ts_sig = pd.Timestamp(ts_override).tz_localize("UTC")
    else:
        ts_sig = _find_latest_feature_ts(cfg["data_root"])
        if ts_sig is None:
            tprint("ERROR: No feature directories found. Run feature_generation first.")
            return

    tprint(f"Labels mode. ts_sig={ts_sig}")

    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])

    # No exchange needed — data already in store, features already on disk
    run_label_generation_step_v2(ts_sig, None, cfg, store, None)

    tprint("LABELS PIPELINE COMPLETE")


def run_features(cfg, ts_override=None):
    if ts_override:
        ts_sig = pd.Timestamp(ts_override).tz_localize("UTC")
    else:
        # Re-use latest existing feature timestamp if available, else current hour
        ts_sig = _find_latest_feature_ts(cfg["data_root"])
        if ts_sig is None:
            ts_sig = pd.Timestamp.utcnow().floor("h")
    tprint(f"Features mode. Target ts_sig={ts_sig}")

    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])

    # Pass None for margin_symbols to trigger auto-refresh in universe logic
    run_feature_generation_step(ts_sig, None, cfg, store)

    tprint("FEATURES PIPELINE COMPLETE")


def run_backtest(cfg, ts_override=None):
    if ts_override:
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


def run_train(cfg, ts_override=None):
    if ts_override:
        ts_sig = pd.Timestamp(ts_override).tz_localize("UTC")
    else:
        ts_sig = _find_latest_feature_ts(cfg["data_root"])
        if ts_sig is None:
            tprint("ERROR: No feature directories found. Run feature_generation first.")
            return

    tprint(f"Train mode. ts_sig={ts_sig}")

    # TP/SL optimisation happens during label generation (see training.generate_label_datasets).
    # Always refresh labels before training so TP:SL widths are re-optimised from current data.
    tprint("Refreshing labels to optimise TP:SL widths before model training (optimise_tpsl_ratio)...")
    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])
    run_label_generation_step_v2(ts_sig, None, cfg, store, None)

    if not _label_artifacts_ready(cfg, ts_sig):
        tprint("ERROR: Label generation did not produce required artifacts. Aborting training.")
        return

    state = run_training_step(ts_sig, cfg, store=store, margin_symbols=None)
    if state:
        tprint("TRAINING PIPELINE COMPLETE")
    else:
        tprint("TRAINING PIPELINE FAILED")


def run_risk_opt(cfg, ts_override=None):
    if ts_override:
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




def run_ridge_sizer(cfg, ts_override=None):
    """Run ridge position sizer on meta model OOF predictions."""
    if ts_override:
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

    tprint(f"Ridge Sizer mode. ts_sig={ts_sig}")
    result = run_ridge_sizer_step(ts_sig, cfg, state_file)
    if result:
        tprint(f"RIDGE SIZER COMPLETE — {len(result.get('weights', {}))} weights learned")
    else:
        tprint("RIDGE SIZER: No results (possibly no meta OOF predictions found)")


def run_all(cfg, ts_override=None):
    """Run download -> features -> train (includes labels) -> ridge_sizer -> optimise in order.
    
    Note: 'train' already refreshes labels internally.
    Note: 'optimise' triggers backtest internally if backtest_results.csv is missing,
          then runs the tpsl_optimiser pipeline (TP/SL calibration, loss limiter,
          profit exit, position sizing, holdout evaluation).
    """
    run_download(cfg)
    run_features(cfg, ts_override=ts_override)
    run_train(cfg, ts_override=ts_override)
    run_ridge_sizer(cfg, ts_override=ts_override)
    run_optimise(cfg, ts_override=ts_override)

    # Final Summary
    if ts_override:
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
                pnl = df["pnl"].sum()
                count = len(df)
                win_rate = (df["pnl"] > 0).mean() if count > 0 else 0.0

                avg_pnl = pnl / count if count > 0 else 0.0

                tprint(f"Total PnL: {pnl:.4f}")
                tprint(f"Total Trades: {count}")
                tprint(f"Win Rate: {win_rate:.2%}")
                tprint(f"Avg PnL per Trade: {avg_pnl:.4f}")
                tprint("==============================\n")
            except Exception as e:
                tprint(f"Could not read results for summary: {e}")


def run_train_meta(cfg, ts_override=None):
    """Re-run only meta model training, reusing existing base models."""
    if ts_override:
        ts_sig = pd.Timestamp(ts_override).tz_localize("UTC")
    else:
        ts_sig = _find_latest_feature_ts(cfg["data_root"])
        if ts_sig is None:
            tprint("ERROR: No feature directories found.")
            return

    from extreme_price_movements.main import train_daily_meta
    store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])
    ex = make_spot_exchange()
    result = train_daily_meta(ts_sig, None, cfg, store, ex)
    if result:
        import pickle as _pkl
        _pkl.dump(result, open("model_state.pkl", "wb"))
        tprint("Meta model state saved to model_state.pkl")
        tprint("TRAIN_META PIPELINE COMPLETE")
    else:
        tprint("TRAIN_META PIPELINE FAILED")


def run_optimise(cfg, ts_override=None):
    if ts_override:
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
    )
    tprint(f"OPTIMISE COMPLETE: {params_path}")

def main():
    parser = argparse.ArgumentParser(description="Extreme Price Movements Pipeline")
    parser.add_argument("mode", choices=["download", "labels", "features", "train", "train_meta", "ridge_sizer", "backtest", "optimize_risk", "optimise", "run"],
                        help="Pipeline mode to run")
    args = parser.parse_args()

    cfg = CFG.copy()

    if args.mode == "download":
        run_download(cfg)
    elif args.mode == "labels":
        run_labels(cfg)
    elif args.mode == "features":
        run_features(cfg)
    elif args.mode == "train":
        run_train(cfg)
    elif args.mode == "train_meta":
        run_train_meta(cfg)
    elif args.mode == "ridge_sizer":
        run_ridge_sizer(cfg)
    elif args.mode == "backtest":
        run_backtest(cfg)
    elif args.mode == "optimize_risk":
        run_risk_opt(cfg)
    elif args.mode == "optimise":
        run_optimise(cfg)
    elif args.mode == "run":
        run_all(cfg)


if __name__ == "__main__":
    main()
