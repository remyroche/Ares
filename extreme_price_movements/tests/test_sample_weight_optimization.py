import numpy as np
import pandas as pd

from extreme_price_movements.purged_cv import IntervalPurgedKFold
from extreme_price_movements.sample_weight_optimization import (
    combine_weights_safely,
    compute_distance_to_barrier_weights,
    compute_liquidity_weights,
    compute_recency_weights,
    compute_vol_weights,
)


def test_combine_weights_safely_does_not_mutate_weights_dict():
    comps = {
        "a": np.array([1.0, 1.2, 0.9, 1.1]),
        "b": np.array([1.0, 1.0, 1.0, 1.0]),  # degenerate should be dropped
    }
    alphas = {"a": 1.0, "b": 1.0}
    out = combine_weights_safely(comps, alphas, min_n_eff_ratio=0.5)
    assert alphas["b"] == 1.0
    assert np.isfinite(out).all()
    assert np.isclose(out.mean(), 1.0, atol=1e-6)


def test_interval_purged_kfold_uses_label_overlap():
    ts = pd.date_range("2025-01-01", periods=20, freq="H")
    intervals = np.column_stack([ts.values, (ts + pd.Timedelta(hours=2)).values])
    cv = IntervalPurgedKFold(n_splits=4, embargo_bars=1)
    for tr, va in cv.split(np.arange(len(ts)), label_intervals=intervals):
        assert len(tr) > 0
        val_start = intervals[va, 0].min()
        val_end = intervals[va, 1].max()
        overlap = (intervals[tr, 0] <= val_end) & (intervals[tr, 1] >= val_start)
        assert not np.any(overlap)


def test_component_weight_functions_are_normalized():
    n = 64
    ts = pd.date_range("2025-01-01", periods=n, freq="H")
    vol = np.linspace(0.5, 1.5, n)
    w_vol = compute_vol_weights(vol, ts.values)
    w_liq = compute_liquidity_weights(np.linspace(100, 1000, n))
    w_rec = compute_recency_weights(np.arange(n), ts.to_period("M").astype(str).values)
    w_dist = compute_distance_to_barrier_weights(
        entry_prices=np.ones(n),
        upper_barriers=np.full(n, 1.03),
        lower_barriers=np.full(n, 0.99),
        atr_past=np.full(n, 0.02),
    )
    for w in [w_vol, w_liq, w_rec, w_dist]:
        assert np.isfinite(w).all()
        assert np.isclose(np.mean(w), 1.0, atol=1e-6)
