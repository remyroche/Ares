import numpy as np
import pytest

from extreme_price_movements.simple_policy_1m_executable_pyramiding import (
    UNFILLED_BAR,
    simulate_executable_pyramiding,
)
from extreme_price_movements.simple_policy_1m_pyramiding_portfolio import (
    allocate_pyramiding_portfolio,
)


def _schedule(*, side=1.0, closes=None, volumes=None, exit_bar=8, x=3, y=0.25):
    closes = np.asarray(
        closes if closes is not None else [[100.0, 101.0, 101.2, 101.3, 101.4, 102.0, 102.1, 102.2, 102.3]],
        dtype=float,
    )
    volumes = np.asarray(volumes if volumes is not None else np.ones_like(closes), dtype=float)
    return simulate_executable_pyramiding(
        np.asarray([0]),
        np.asarray([100.0]),
        closes,
        volumes,
        np.asarray([side]),
        np.asarray([0.01]),
        np.asarray([10.0]),
        np.asarray([20.0]),
        np.asarray([exit_bar]),
        np.asarray([103.0 if side > 0 else 97.0]),
        np.ones(x),
        y,
        0.005,
        minimum_bars_between_fills=5,
        minimum_gap_bps=50.0,
    )


def test_executable_schedule_enforces_cooldown_and_one_fill_per_bar():
    out = _schedule()
    bars = out["fill_bars"][0, : out["filled_tranche_count"][0]]
    assert bars.tolist() == [-1, 4]
    assert np.diff(bars).min() >= 5
    assert len(np.unique(bars)) == len(bars)


def test_zero_volume_and_exit_bar_cannot_fill():
    closes = [[100, 101, 101, 101, 101, 101, 101, 101, 102]]
    volumes = [[1, 1, 1, 1, 0, 0, 0, 0, 1]]
    out = _schedule(closes=closes, volumes=volumes, exit_bar=8)
    assert out["filled_tranche_count"][0] == 1
    assert out["zero_volume_rejections"][0] > 0
    assert out["exit_bar_collisions"][0] == 1
    assert out["fill_bars"][0, 1] == UNFILLED_BAR


@pytest.mark.parametrize("side", [1.0, -1.0])
def test_actual_fill_gap_respects_full_spread_floor(side):
    if side > 0:
        closes = [[100, 100.2, 100.3, 100.4, 100.6, 101.2, 101.3, 101.4, 101.5]]
    else:
        closes = [[100, 99.8, 99.7, 99.6, 99.4, 98.8, 98.7, 98.6, 98.5]]
    out = _schedule(side=side, closes=closes)
    raw = out["fill_raw_prices"][0, : out["filled_tranche_count"][0]]
    assert len(raw) >= 2
    gaps_bps = side * np.diff(raw) / raw[:-1] * 10_000
    assert np.all(gaps_bps >= 50.0 - 1e-9)


def test_portfolio_allocator_enforces_book_and_position_caps():
    n = 3
    fill_bars = np.full((n, 8), -2, dtype=np.int32)
    fill_bars[:, 0] = -1
    fill_bars[:, 1] = 4
    returns = np.full((n, 8), np.nan)
    returns[:, :2] = 0.02
    out = allocate_pyramiding_portfolio(
        np.asarray([0, 0, 0], dtype=np.int64),
        np.asarray([0, 1, 2], dtype=np.int32),
        np.asarray([0.99, 0.98, 0.97]),
        np.asarray([10, 10, 10], dtype=np.int32),
        fill_bars,
        returns,
        returns,
        np.asarray([0.20, 0.20, 0.20]),
        np.asarray([True, True, True]),
        np.asarray([0.5, 0.5]),
        wallet_cap=0.25,
        position_cap=0.15,
        max_open=8,
        max_new_per_minute=2,
        max_dca_per_minute=2,
        dca_priority_bonus=0.0,
        minimum_order=0.001,
    )
    selected, allocated, diagnostics = out[0], out[3], out[-1]
    assert selected.sum() == 2
    assert allocated.max() <= 0.15 + 1e-12
    assert diagnostics[0] <= 0.25 + 1e-12
