import numpy as np
import pandas as pd

from extreme_price_movements.simple_policy_1m_sizing_normalizers import (
    archetype_ewma_normalize,
    bounded_dynamic_exposure_normalize,
    open_portfolio_budget_normalize,
    rolling_window_normalize,
    selected_trades,
)


def _fixture():
    rows = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 00:00Z", "2026-01-01 01:00Z", "2026-01-01 02:00Z"]),
        "symbol": ["A", "B", "C"], "rank_pct": [1.0, 1.0, 1.0],
        "policy_archetype": ["x", "x", "y"],
    })
    outputs = {"exit_bars": np.array([30, 30, 30]), "net_return": np.ones(3)}
    return rows, outputs


def test_selected_and_bounded_dynamic_exposure_are_causal_and_bounded():
    rows, outputs = _fixture()
    chosen = selected_trades(rows, outputs)
    assert len(chosen.local_index) == 3
    raw = np.array([1.2, 1.2, 1.2])
    adjusted = bounded_dynamic_exposure_normalize(rows, outputs, raw, exposure_band=0.05)
    assert np.allclose(adjusted, 1.05)


def test_rolling_and_archetype_ewma_use_prior_fit_state():
    fit_rows, fit_outputs = _fixture()
    apply_rows = fit_rows.copy(); apply_rows["timestamp"] += pd.Timedelta(days=1)
    fit_size = np.array([1.2, 1.2, 0.8]); apply_size = np.array([1.2, 1.2, 0.8])
    rolling = rolling_window_normalize(fit_rows, fit_outputs, fit_size, apply_rows, fit_outputs, apply_size, window_hours=72)
    assert rolling[0] < 1.2
    ewma = archetype_ewma_normalize(fit_rows, fit_outputs, fit_size, apply_rows, fit_outputs, apply_size, half_life_hours=24)
    assert np.allclose(ewma, 1.0, atol=1e-8)


def test_rolling_normalizer_does_not_use_later_or_same_timestamp_entries():
    fit_rows, fit_outputs = _fixture()
    apply_rows = fit_rows.copy(); apply_rows["timestamp"] += pd.Timedelta(days=1)
    apply_rows.loc[1, "timestamp"] = apply_rows.loc[0, "timestamp"]
    first = np.array([1.2, 0.7, 1.4])
    changed_future = np.array([1.2, 1.5, 0.5])
    a = rolling_window_normalize(fit_rows, fit_outputs, np.ones(3), apply_rows, fit_outputs, first, window_hours=72)
    b = rolling_window_normalize(fit_rows, fit_outputs, np.ones(3), apply_rows, fit_outputs, changed_future, window_hours=72)
    assert np.isclose(a[0], b[0])


def test_open_portfolio_budget_tracks_baseline_when_unclipped():
    rows, outputs = _fixture()
    raw = np.array([1.2, 1.2, 1.2])
    adjusted = open_portfolio_budget_normalize(rows, outputs, raw, lower=0.1, upper=2.0)
    assert np.allclose(adjusted, 1.0)
