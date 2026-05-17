from __future__ import annotations

import time

import numpy as np
from scipy.stats import norm

import extreme_price_movements.fast_funcs as ff
from extreme_price_movements.inference.live_zscore_state import RollingZScoreState


def _time_call(label, fn):
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    print(f"{label}: {dt:.4f}s", flush=True)
    return result, dt


def _bench_case(rows: int, cols: int, window: int) -> None:
    rng = np.random.default_rng(42)
    mat = rng.normal(size=(rows, cols)).astype(np.float32)
    mat[::997, ::11] = np.nan

    # Compile outside timed region.
    ff._numba_rolling_zscore_parallel(mat[:1024, : min(cols, 8)], window)
    ff._numba_rolling_zscore_fused_parallel(mat[:1024, : min(cols, 8)], window)

    print(f"\ncase: T={rows:,} C={cols:,} window={window}", flush=True)
    old, _ = _time_call("old parallel z-score", lambda: ff._numba_rolling_zscore_parallel(mat, window))
    fused, _ = _time_call("fused z-score", lambda: ff._numba_rolling_zscore_fused_parallel(mat, window))
    diff = np.nanmax(np.abs(old.astype(np.float64) - fused.astype(np.float64)))
    print(f"max abs diff vs old: {diff:.8g}", flush=True)

    keys = [f"ret24h_{i}" for i in range(cols)]
    symbols = ["SYM"]
    state = RollingZScoreState(
        keys,
        symbols,
        window=window,
        winsor_qt=0.02,
        sigma_k=float(norm.ppf(0.98)),
    )
    seed_rows = min(rows - 1, window)
    for i in range(seed_rows):
        state.update({keys[j]: np.asarray([mat[i, j]], dtype=np.float32) for j in range(cols)})
    _, live_dt = _time_call(
        "live stateful single-row update",
        lambda: state.update(
            {keys[j]: np.asarray([mat[seed_rows, j]], dtype=np.float32) for j in range(cols)}
        ),
    )
    print(f"live stateful per feature-symbol: {live_dt / cols * 1e6:.3f}us", flush=True)


def main() -> None:
    window = 720
    _bench_case(100_000, 50, window)
    _bench_case(100_000, 200, window)


if __name__ == "__main__":
    main()
