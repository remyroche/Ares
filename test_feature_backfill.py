#!/usr/bin/env python3
"""Quick test: run feature backfill on 3 symbols only."""
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_epm")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd
from extreme_price_movements.config import CFG
from extreme_price_movements.data_store import PartitionedOHLCVStore
from extreme_price_movements.pipeline_steps import run_feature_generation_step
from extreme_price_movements.utils import tprint

cfg = CFG.copy()
# Normalise paths
for k in ("data_root", "reports_root", "hf_data_dir"):
    if k in cfg:
        cfg[k] = os.path.join(parent_dir, cfg[k])

ts_sig = pd.Timestamp("2026-03-21 14:00:00", tz="UTC")
store = PartitionedOHLCVStore(root_dir=cfg["data_root"], timeframe=cfg["timeframe"])

# Override market_basket to just 3 symbols for quick test
test_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
cfg["market_basket"] = test_symbols
cfg["fetch_symbols_M"] = 3  # only top-3

tprint(f"Test backfill: {test_symbols} @ {ts_sig}")
run_feature_generation_step(
    ts_sig,
    test_symbols,  # pass explicit symbols
    cfg,
    store,
    force_full_recompute=False,
)
tprint("TEST BACKFILL COMPLETE")
