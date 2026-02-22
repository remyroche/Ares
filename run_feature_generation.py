#!/usr/bin/env python3
"""
Run extreme_price_movements feature generation step
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, '/Users/remyroche/Documents/Ares')

from extreme_price_movements.config import CFG
from extreme_price_movements.data_store import make_spot_exchange, PartitionedOHLCVStore
from extreme_price_movements.universe import refresh_margin_universe_daily
from extreme_price_movements.pipeline_steps import run_feature_generation_step
from extreme_price_movements.utils import tprint, Timer

def main():
    tprint("Starting extreme_price_movements feature generation...")
    
    # Initialize exchange and store
    ex = make_spot_exchange()
    store = PartitionedOHLCVStore(exchange=ex, data_root=CFG["data_root"])
    
    # Get margin symbols
    margin_symbols = refresh_margin_universe_daily(ex, top_M=CFG["fetch_symbols_M"])
    tprint(f"Margin universe: {len(margin_symbols)} symbols")
    
    # Use current timestamp or most recent hour
    ts_sig = datetime.now().floor('H') - pd.Timedelta(hours=1)  # Previous hour to ensure data availability
    tprint(f"Target timestamp: {ts_sig}")
    
    # Run feature generation
    with Timer("Feature Generation Step"):
        run_feature_generation_step(
            ts_sig=ts_sig,
            margin_symbols=margin_symbols,
            cfg=CFG,
            store=store,
            force_full_recompute=False  # Set to True to force full recompute
        )
    
    tprint("Feature generation completed successfully!")

if __name__ == "__main__":
    main()
