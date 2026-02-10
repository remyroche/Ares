import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from extreme_price_movements.features import compute_features_hourly
from extreme_price_movements.config import CFG

def generate_data(n_symbols=2, n_rows=200):
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="1h")
    symbols = [f"SYM{i}" for i in range(n_symbols)]

    panel = {}
    np.random.seed(42)
    for s in symbols:
        returns = np.random.normal(0, 0.01, size=len(dates))
        price = 100 * np.exp(np.cumsum(returns))
        high = price * (1 + np.abs(np.random.normal(0, 0.005, size=len(dates))))
        low = price * (1 - np.abs(np.random.normal(0, 0.005, size=len(dates))))
        close = price
        open_p = price * (1 + np.random.normal(0, 0.002, size=len(dates)))
        volume = np.random.lognormal(10, 1, size=len(dates)) # Raw volume ~22000

        if "close" not in panel:
            panel["close"] = pd.DataFrame(index=dates, columns=symbols)
            panel["open"] = pd.DataFrame(index=dates, columns=symbols)
            panel["high"] = pd.DataFrame(index=dates, columns=symbols)
            panel["low"] = pd.DataFrame(index=dates, columns=symbols)
            panel["volume"] = pd.DataFrame(index=dates, columns=symbols)

        panel["close"][s] = close.astype(np.float32)
        panel["open"][s] = open_p.astype(np.float32)
        panel["high"][s] = high.astype(np.float32)
        panel["low"][s] = low.astype(np.float32)
        panel["volume"][s] = volume.astype(np.float32)

    mkt_gates = pd.DataFrame(index=dates)
    mkt_gates["mkt_rv_ratio"] = 1.0
    mkt_gates["mkt_rv_pct"] = 0.5
    mkt_gates["abs_mkt_ret24h_z"] = 0.0
    mkt_gates["trend_bin3"] = 0
    mkt_gates["mkt_trend"] = 0.0
    mkt_gates["mkt_rv"] = 0.01
    mkt_gates["mkt_close"] = 100.0
    mkt_gates["mkt_ret6h"] = 0.0
    for c in mkt_gates.columns:
        mkt_gates[c] = mkt_gates[c].astype(np.float32)

    return panel, mkt_gates

def inspect_values():
    print("Generating data...")
    panel, mkt_gates = generate_data()

    print("Computing features...")
    # Override cache to force computation
    from extreme_price_movements import features
    features._cache.clear()

    feats = compute_features_hourly(panel, mkt_gates, CFG)

    # Inspect v-related features
    print("\n--- Feature Inspection ---")

    # Check if 'vw_breakout' has Inf
    vw = feats.get("vw_breakout")
    if vw is not None:
        print(f"vw_breakout: Min={vw.min().min()}, Max={vw.max().max()}, HasInf={np.isinf(vw).any().any()}")
        if np.isinf(vw).any().any():
            print("WARNING: vw_breakout contains Inf! Volume assumption likely wrong.")

    # We can't see 'v' directly but 'vol_z' or 'rvol_z' gives clues.
    # Also 'rvol_ratio' isn't saved, but 'vw_breakout' = breakout_z * log(1+rvol).
    # If rvol exploded, vw_breakout would be huge or Inf (if log overflowed? No log grows slowly).
    # Wait, rvol_ratio = exp(v - ema). If v is raw, this is exp(large) = Inf.
    # log(1 + Inf) = Inf.
    # So Inf in vw_breakout confirms raw volume hypothesis.

    # Also check 'vol_price_spread' = v / (h-l).
    # If v is log (~10) and h-l is FFD price range (~0.01), ratio is ~1000.
    # If v is raw (~22000), ratio is ~2,000,000.
    vps = feats.get("vol_price_spread")
    if vps is not None:
        print(f"vol_price_spread: Mean={vps.mean().mean()}")

    # Check vol_price_div
    vpd = feats.get("vol_price_div")
    if vpd is not None:
        print(f"vol_price_div: Mean={vpd.mean().mean()}")

if __name__ == "__main__":
    inspect_values()
