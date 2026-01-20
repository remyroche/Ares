import numpy as np
from src.utils.layer5_optimized import apply_position_sizing_numba, run_atr_backtest_numba
import time

def test_funcs():
    print("Generating data...")
    n = 1000
    prices = np.cumprod(1 + np.random.normal(0, 0.01, n)) * 100
    highs = prices * (1 + np.abs(np.random.normal(0, 0.005, n)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.005, n)))
    atr = prices * 0.02
    probs = np.random.uniform(0, 1, n)
    dampening = np.random.uniform(0, 1, n)

    print("Testing sizing...")
    sizes = apply_position_sizing_numba(
        probs, dampening,
        threshold=0.5, kelly_fraction=0.1, steepness=2.0, dampening_mult=0.5
    )
    print(f"Sizes range: {sizes.min()} - {sizes.max()}")

    print("Testing backtest...")
    t0 = time.time()
    equity, trades = run_atr_backtest_numba(
        prices, highs, lows, atr, sizes,
        sl_atr_mult=2.0, trail_trigger_mult=2.0, trail_dist_mult=1.0
    )
    t1 = time.time()
    print(f"Backtest took {t1-t0:.4f}s")
    print(f"Equity end: {equity[-1]}")
    print(f"Trades count: {len(trades)}")
    if len(trades) > 0:
        print(f"Sample trade: {trades[0]}")

if __name__ == "__main__":
    test_funcs()
