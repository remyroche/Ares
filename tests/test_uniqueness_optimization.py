import time

import numpy as np
import pandas as pd

from src.training.steps.labeling.generate_weights_per_label import compute_uniqueness_weights as compute_uniqueness_vectorized


def compute_uniqueness_loop(
    t1: pd.Series,
    events_index: pd.DatetimeIndex | None = None,
    market_index: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Reference implementation matching the original loop-based logic.

    This is used only for verification of the optimized vectorized implementation.
    """
    if len(t1) == 0:
        return pd.Series(dtype=float)

    # 1. Expand events to valid time range
    if events_index is not None:
        t1_aligned = pd.Series(t1.values, index=events_index)
    else:
        t1_aligned = t1.copy()

    t1_aligned = t1_aligned.sort_index()

    # 2. Derive concurrency (how many events are active at each time step)
    start_times = pd.Series(1, index=t1_aligned.index)
    end_times = pd.Series(-1, index=t1_aligned.values)

    timeline = pd.concat([start_times, end_times]).sort_index()
    timeline = timeline.groupby(timeline.index).sum()
    concurrency = timeline.cumsum()

    # 3. Optionally align to a provided market index
    if market_index is not None:
        concurrency_aligned = concurrency.reindex(market_index, method="ffill").fillna(0)
    else:
        concurrency_aligned = concurrency

    weights = pd.Series(index=t1_aligned.index, dtype=float)

    for t0, t_end in t1_aligned.items():
        event_concurrency = concurrency_aligned.loc[t0:t_end]

        if len(event_concurrency) > 0:
            avg_uniqueness = (1.0 / np.maximum(1, event_concurrency)).mean()
            weights[t0] = avg_uniqueness
        else:
            weights[t0] = 1.0

    if weights.sum() > 0:
        weights *= len(weights) / weights.sum()

    return weights


def _make_synthetic_events(
    n_ticks: int = 50_000,
    n_events: int = 10_000,
    freq: str = "T",
) -> tuple[pd.Series, pd.DatetimeIndex]:
    """Generate synthetic overlapping events on a regular time grid."""
    market_index = pd.date_range("2020-01-01", periods=n_ticks, freq=freq)

    rng = np.random.RandomState(42)
    start_indices = rng.randint(0, n_ticks - 10, size=n_events)
    durations = rng.randint(1, 50, size=n_events)
    end_indices = np.minimum(start_indices + durations, n_ticks - 1)

    t0 = market_index[start_indices]
    t1 = market_index[end_indices]

    # Ensure no duplicate start indices for indexing; if duplicates exist, aggregate
    t1_series = pd.Series(t1, index=t0).sort_index()
    # If there are duplicate indices, keep the max end (longest event) for determinism
    t1_series = t1_series.groupby(t1_series.index).max()

    return t1_series, market_index


def run_equivalence_checks() -> None:
    t1_series, market_index = _make_synthetic_events()

    # 1) Path without explicit market_index
    w_loop_no_mkt = compute_uniqueness_loop(t1_series, events_index=None, market_index=None)
    w_vec_no_mkt = compute_uniqueness_vectorized(t1_series, events_index=None, market_index=None)

    assert w_loop_no_mkt.index.equals(w_vec_no_mkt.index)
    diff_no_mkt = np.max(np.abs(w_loop_no_mkt.values - w_vec_no_mkt.values))

    # 2) Path with explicit market_index
    w_loop_mkt = compute_uniqueness_loop(t1_series, events_index=None, market_index=market_index)
    w_vec_mkt = compute_uniqueness_vectorized(t1_series, events_index=None, market_index=market_index)

    assert w_loop_mkt.index.equals(w_vec_mkt.index)
    diff_mkt = np.max(np.abs(w_loop_mkt.values - w_vec_mkt.values))

    tol = 1e-10
    assert diff_no_mkt <= tol, f"Vectorized vs loop (no market_index) max diff {diff_no_mkt} > {tol}"
    assert diff_mkt <= tol, f"Vectorized vs loop (market_index) max diff {diff_mkt} > {tol}"

    print("Equivalence checks passed.")
    print(f"  Max abs diff (no market_index): {diff_no_mkt:.3e}")
    print(f"  Max abs diff (with market_index): {diff_mkt:.3e}")


def run_benchmarks() -> None:
    # Use a larger problem size for timing
    t1_series, market_index = _make_synthetic_events(n_ticks=100_000, n_events=20_000)

    # Warm-up
    compute_uniqueness_loop(t1_series, events_index=None, market_index=market_index)
    compute_uniqueness_vectorized(t1_series, events_index=None, market_index=market_index)

    n_runs = 3

    # Loop-based timing
    t0 = time.perf_counter()
    for _ in range(n_runs):
        compute_uniqueness_loop(t1_series, events_index=None, market_index=market_index)
    t_loop = (time.perf_counter() - t0) / n_runs

    # Vectorized timing
    t0 = time.perf_counter()
    for _ in range(n_runs):
        compute_uniqueness_vectorized(t1_series, events_index=None, market_index=market_index)
    t_vec = (time.perf_counter() - t0) / n_runs

    speedup = t_loop / t_vec if t_vec > 0 else float("inf")

    print("Benchmark results (average over", n_runs, "runs):")
    print(f"  Loop-based:      {t_loop:.4f} s")
    print(f"  Vectorized:      {t_vec:.4f} s")
    print(f"  Speedup factor:  {speedup:.2f}x")


if __name__ == "__main__":
    print("Running uniqueness equivalence checks and benchmarks...")
    run_equivalence_checks()
    run_benchmarks()
