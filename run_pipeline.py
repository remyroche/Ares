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
from extreme_price_movements.time_utils import get_ts_sig

def main():
    parser = argparse.ArgumentParser(description="Extreme Price Movements Pipeline")
    parser.add_argument("--light", action="store_true", help="Run in light mode (less data)")
    parser.add_argument("--mode", choices=["download", "feature_generation", "train", "run"], required=True, help="Pipeline mode")
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
            except Exception as e:
                tprint(f"Error fetching {s}: {e}")
        tprint("Download Complete.")

    elif args.mode == "feature_generation":
        tprint("Starting Feature Generation...")
        with Timer("Margin Universe"):
            mu = refresh_margin_universe_daily(None, quote="USDT")

        ts_sig = get_ts_sig()
        generate_features_daily(ts_sig, mu.symbols, CFG, store, ex)

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
