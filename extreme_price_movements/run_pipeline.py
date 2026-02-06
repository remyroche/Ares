#!/usr/bin/env python3
"""
CLI entry point for extreme_price_movements pipeline.

Usage:
    python3 extreme_price_movements/run_pipeline.py labels
"""
import sys
import argparse
import pandas as pd

from extreme_price_movements.config import CFG
from extreme_price_movements.utils import tprint
from extreme_price_movements.data_store import PartitionedOHLCVStore
from extreme_price_movements.pipeline_steps import (
    run_label_generation_step_v2,
    run_feature_generation_step,
    run_training_step,
    run_backtest_step,
    run_risk_optimization_step
)


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




def _label_artifacts_ready(cfg, ts_sig):
    """Check whether core label artifacts exist for this run timestamp."""
    import os
    run_id = ts_sig.strftime("%Y%m%d_%H%M%S")
    horizons = cfg.get("label_horizons_hours", [])
    required = [
        "spike_anatomy_best",
        "spike_anatomy_worst",
        "exhaustion_history",
        "exh_up",
        "exh_down",
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

    state = run_training_step(ts_sig, cfg)
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




def run_all(cfg, ts_override=None):
    """Run features -> labels -> train -> risk optimisation -> backtest in order."""
    run_features(cfg, ts_override=ts_override)
    run_labels(cfg, ts_override=ts_override)
    run_train(cfg, ts_override=ts_override)
    run_risk_opt(cfg, ts_override=ts_override)
    run_backtest(cfg, ts_override=ts_override)

def main():
    parser = argparse.ArgumentParser(description="Extreme Price Movements Pipeline")
    parser.add_argument("mode", choices=["labels", "features", "train", "backtest", "optimize_risk", "run"],
                        help="Pipeline mode to run")
    args = parser.parse_args()

    cfg = CFG.copy()

    if args.mode == "labels":
        run_labels(cfg)
    elif args.mode == "features":
        run_features(cfg)
    elif args.mode == "train":
        run_train(cfg)
    elif args.mode == "backtest":
        run_backtest(cfg)
    elif args.mode == "optimize_risk":
        run_risk_opt(cfg)
    elif args.mode == "run":
        run_all(cfg)


if __name__ == "__main__":
    main()
