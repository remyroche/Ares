import argparse
import sys
import pandas as pd
import time
from extreme_price_movements.config import CFG, update_config_for_mode
from extreme_price_movements.main import run_live_cycle, train_daily
from extreme_price_movements.data_store import make_spot_exchange, PartitionedOHLCVStore
from extreme_price_movements.universe import refresh_margin_universe_daily, build_fetch_universe
from extreme_price_movements.time_utils import get_ts_sig
from extreme_price_movements.utils import tprint

def main():
    parser = argparse.ArgumentParser(description="Extreme Price Movements Strategy CLI")
    parser.add_argument("command", choices=["download", "train", "trade"], help="Command to execute")
    parser.add_argument("--mode", choices=["standard", "light"], default="standard", help="Run mode (standard=4y, light=0.5y)")
    args = parser.parse_args()

    tprint(f"CLI: {args.command} mode={args.mode}")
    update_config_for_mode(args.mode)

    ex = make_spot_exchange()
    store = PartitionedOHLCVStore(root_dir=CFG["data_root"], timeframe=CFG["timeframe"])

    if args.command == "download":
        tprint("Starting data download...")
        mu = refresh_margin_universe_daily(None, quote="USDT")

        syms_all = build_fetch_universe(mu.symbols, CFG["market_basket"], CFG["fetch_symbols_M"])

        ts_sig = get_ts_sig()
        days = int(CFG["fetch_years"] * 365)
        since = (ts_sig - pd.Timedelta(days=days)).floor("D")
        since_ms = int(since.value // 10**6)

        tprint(f"Fetching {len(syms_all)} symbols since {since}...")
        for i, s in enumerate(syms_all):
             if i % 10 == 0: tprint(f"Progress {i}/{len(syms_all)}")
             try: store.update_symbol(ex, s, since_ms)
             except Exception as e: tprint(f"Error fetching {s}: {e}")
        tprint("Download complete.")

    elif args.command == "train":
        tprint("Starting training...")
        mu = refresh_margin_universe_daily(None, quote="USDT")
        ts_sig = get_ts_sig()
        train_daily(ts_sig, mu.symbols, CFG, store, ex)

    elif args.command == "trade":
        tprint("Starting live trading loop...")
        while True:
            try:
                run_live_cycle()
            except Exception as e:
                tprint(f"CRITICAL ERROR: {e}")
                import traceback; traceback.print_exc()
            tprint("Sleeping 60s...")
            time.sleep(60)

if __name__ == "__main__":
    main()
