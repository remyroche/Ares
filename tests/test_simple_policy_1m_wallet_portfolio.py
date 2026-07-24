import numpy as np

from extreme_price_movements.simple_policy_1m_wallet_portfolio import (
    replay_marked_notional_wallet,
)


def _replay(*, closes, sides=None, fractions=None, exits=None, returns=None):
    n = len(closes)
    return replay_marked_notional_wallet(
        timestamps_ns=np.arange(n, dtype=np.int64) * 60 * 1_000_000_000,
        symbol_codes=np.arange(n, dtype=np.int32),
        side=np.ones(n) if sides is None else np.asarray(sides, dtype=float),
        raw_entry_prices=np.full(n, 100.0),
        entry_half_spread_bps=np.zeros(n),
        close_paths=np.asarray(closes, dtype=float),
        exit_bars=np.full(n, 5, dtype=np.int32) if exits is None else np.asarray(exits),
        net_returns=np.zeros(n) if returns is None else np.asarray(returns, dtype=float),
        requested_fractions=(
            np.full(n, 0.50) if fractions is None else np.asarray(fractions, dtype=float)
        ),
        max_wallet_invested=0.80,
        max_new_per_bar=2,
    )


def test_wallet_replay_clips_second_entry_to_80pct_without_count_cap():
    closes = np.full((2, 8), 100.0)
    result = _replay(closes=closes)
    assert result["selected"].tolist() == [True, True]
    np.testing.assert_allclose(result["admitted_notional"], [0.5, 0.3], atol=1e-12)


def test_wallet_replay_marks_long_and_short_gross_without_netting():
    closes = np.array(
        [
            [110.0, 110.0, 110.0],
            [90.0, 90.0, 90.0],
            [100.0, 100.0, 100.0],
        ]
    )
    result = _replay(
        closes=closes,
        sides=[1.0, -1.0, 1.0],
        fractions=[0.30, 0.30, 0.30],
        exits=[5, 5, 5],
    )
    # Dynamic equity sizes the second entry at 0.309. At t=2 the long and short
    # are 0.33 / 0.2781 marked notional and remain gross rather than netted.
    np.testing.assert_allclose(result["marked_notional_before"][2], 0.6081, atol=1e-12)
    np.testing.assert_allclose(result["admitted_notional"][2], 0.24062, atol=1e-12)


def test_wallet_replay_allows_more_than_eight_small_positions():
    n = 10
    closes = np.full((n, 20), 100.0)
    result = _replay(
        closes=closes,
        fractions=np.full(n, 0.05),
        exits=np.full(n, 15),
    )
    assert int(result["selected"].sum()) == 10
    np.testing.assert_allclose(result["admitted_notional"].sum(), 0.50, atol=1e-12)
