import argparse
import pickle
import os
import sys
import pandas as pd
from extreme_price_movements.config import CFG
from extreme_price_movements.utils import tprint, Timer
from extreme_price_movements.data_store import make_spot_exchange, PartitionedOHLCVStore
from extreme_price_movements.universe import refresh_margin_universe_daily, build_fetch_universe
from extreme_price_movements.main import train_daily, run_live_cycle, generate_features_daily
from extreme_price_movements.pipeline_steps import run_label_generation_step_v2, run_risk_optimization_step, run_backtest_step
from extreme_price_movements.time_utils import get_ts_sig

def validate_downloaded_data(store, symbol):
    safe_sym = symbol.replace("/", "_")
    sym_dir = os.path.join(store.ohlcv_dir, f"symbol={safe_sym}")

    if not os.path.isdir(sym_dir):
        tprint(f"WARNING: No data directory for {symbol} (Validation skipped)")
        return

    files = []
    for root, _, filenames in os.walk(sym_dir):
        for f in filenames:
            if f.endswith(".parquet"):
                files.append(os.path.join(root, f))

    if not files:
        tprint(f"WARNING: No parquet files found for {symbol}")
        return

    total_rows = 0
    required_cols = ["open", "high", "low", "close", "volume"]
    files_checked = 0

    for fpath in files:
        try:
            df = pd.read_parquet(fpath)
        except Exception as e:
            raise RuntimeError(f"Failed to read {fpath}: {e}")

        # 1. Check columns
        if not all(col in df.columns for col in required_cols):
            missing = list(set(required_cols) - set(df.columns))
            raise ValueError(f"Missing columns {missing} in {fpath}")

        # 2. Check not empty
        if df.empty:
            raise ValueError(f"File is empty: {fpath}")

        # 3. Verify 10 random rows
        n_sample = min(10, len(df))
        sample = df.sample(n=n_sample)

        # Check for NaNs in required columns
        if sample[required_cols].isnull().any().any():
             raise ValueError(f"Found NaNs in sample of {fpath}")

        total_rows += len(df)
        files_checked += 1

    tprint(f"Validated {symbol}: {files_checked} files, {total_rows} rows. OK.")

def main():
    parser = argparse.ArgumentParser(description="Extreme Price Movements Pipeline")
    parser.add_argument("--light", action="store_true", help="Run in light mode (less data)")
    parser.add_argument("--mode", choices=["download", "feature_generation", "labels", "train", "risk", "backtest", "run"], required=True, help="Pipeline mode")
    parser.add_argument("--state-file", default="model_state.pkl", help="Path to persist model state")

    args = parser.parse_args()

    if args.light:
        CFG["fetch_years"] = 0.5
        tprint("LIGHT MODE ENABLED: fetch_years set to 0.5")

    ex = make_spot_exchange()
    store = PartitionedOHLCVStore(root_dir=CFG["data_root"], timeframe=CFG["timeframe"])

    if args.mode == "download":
        tprint("Starting Data Download...")
        with Timer("Margin Universe"):
            mu = refresh_margin_universe_daily(None, quote="USDT")

        syms_all = build_fetch_universe(mu.symbols, CFG["market_basket"], CFG["fetch_symbols_M"])
        tprint(f"Universe size: {len(syms_all)}")

        ts_now = pd.Timestamp.utcnow()
        days = int(CFG["fetch_years"] * 365)
        since = (ts_now - pd.Timedelta(days=days)).floor("D")
        since_ms = int(since.value // 10**6)

        tprint(f"Fetching data since {since}...")
        for i, s in enumerate(syms_all):
            try:
                tprint(f"Downloading {i+1}/{len(syms_all)}: {s}...")
                store.update_symbol(ex, s, since_ms)
                validate_downloaded_data(store, s)
            except Exception as e:
                tprint(f"Error fetching {s}: {e}")
        tprint("Download Complete.")

    elif args.mode == "feature_generation":
        tprint("Starting Feature Generation...")
        with Timer("Margin Universe"):
            mu = refresh_margin_universe_daily(None, quote="USDT")

        ts_sig = get_ts_sig()
        generate_features_daily(ts_sig, mu.symbols, CFG, store, ex)

    elif args.mode == "labels":
        tprint("Starting Label Generation...")
        with Timer("Margin Universe"):
            mu = refresh_margin_universe_daily(None, quote="USDT")

        ts_sig = get_ts_sig()
        run_label_generation_step_v2(ts_sig, mu.symbols, CFG, store, ex)

    elif args.mode == "train":
        tprint("Starting Training...")
        with Timer("Margin Universe"):
            mu = refresh_margin_universe_daily(None, quote="USDT")

        ts_sig = get_ts_sig()

        state = train_daily(ts_sig, mu.symbols, CFG, store, ex)
        if state:
            with open(args.state_file, "wb") as f:
                pickle.dump(state, f)
            tprint(f"Model state saved to {args.state_file}")
        else:
            tprint("Training failed or produced no state.")

    elif args.mode == "risk":
        tprint("Starting Risk Optimization...")
        with Timer("Margin Universe"):
            mu = refresh_margin_universe_daily(None, quote="USDT")

        ts_sig = get_ts_sig()
        run_risk_optimization_step(ts_sig, mu.symbols, CFG, store, args.state_file)

    elif args.mode == "backtest":
        tprint("Starting Backtest...")
        with Timer("Margin Universe"):
            mu = refresh_margin_universe_daily(None, quote="USDT")

        ts_sig = get_ts_sig()
        run_backtest_step(ts_sig, mu.symbols, CFG, store, args.state_file)

    elif args.mode == "run":
        tprint("Starting Live Cycle...")
        loaded_state = None
        if os.path.exists(args.state_file):
            try:
                tprint(f"Loading initial state from {args.state_file}")
                with open(args.state_file, "rb") as f:
                    loaded_state = pickle.load(f)
            except Exception as e:
                tprint(f"Error loading state: {e}")

        run_live_cycle(initial_model_state=loaded_state)

if __name__ == "__main__":
    main()
