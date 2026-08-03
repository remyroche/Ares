from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.simple_policy_optimiser import simulate_and_score


def _row(side: float, *, spread_bps: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"]),
            "symbol": ["TEST/USD:USD"],
            "rank_pct": [1.0],
            "side": [side],
            "barrier_pct": [0.01],
            "expected_spread_bps": [spread_bps],
        }
    )


def _common() -> dict[str, float | int]:
    return {
        "sl_mult": 10.0,
        "trailing_activation_mult": 1.0,
        "trailing_power": 1.5,
        "trailing_squash_divisor": 2.0,
        "giveback_beta": 0.5,
        "capital_protect_mfe_mult": 0.0,
        "adverse_exit_enabled": False,
        "max_concurrent_trades": 10,
        "max_concurrent_per_asset": 10,
        "max_new_entries_per_bar": 10,
    }


def test_partial_exit_disabled_is_exact_baseline() -> None:
    paths = (
        np.array([[100.0, 100.0, 102.0, 102.0]], dtype=np.float32),
        np.array([[100.0, 102.0, 103.0, 102.5]], dtype=np.float32),
        np.array([[100.0, 99.8, 101.5, 101.0]], dtype=np.float32),
        np.array([[100.0, 101.8, 102.5, 101.5]], dtype=np.float32),
    )
    baseline = simulate_and_score(
        _row(1.0), *paths, cost_pct=0.001, **_common()
    )
    disabled = simulate_and_score(
        _row(1.0),
        *paths,
        cost_pct=0.001,
        partial_exit_on_first_trailing_activation_fraction=0.0,
        **_common(),
    )
    for field in (
        "raw_gains",
        "gross_gains",
        "exit_bars",
        "entry_prices",
        "exit_prices",
        "gross_returns",
        "fee_returns",
        "net_returns",
        "selected_mask",
    ):
        np.testing.assert_array_equal(baseline[field], disabled[field])
    assert baseline["exit_reason"] == disabled["exit_reason"]
    np.testing.assert_array_equal(
        disabled["p50_gross_returns"], disabled["gross_returns"]
    )
    np.testing.assert_array_equal(
        disabled["p50_fee_returns"], disabled["fee_returns"]
    )
    np.testing.assert_array_equal(
        disabled["p50_net_returns"], disabled["net_returns"]
    )
    assert not disabled["partial_exit_mask"].any()


@pytest.mark.parametrize(
    ("side", "opens", "highs", "lows"),
    [
        (
            1.0,
            [100.0, 100.0, 102.0, 103.0, 102.0],
            [100.0, 102.0, 103.0, 103.5, 102.5],
            [100.0, 99.8, 101.5, 102.0, 101.0],
        ),
        (
            -1.0,
            [100.0, 100.0, 98.0, 97.0, 98.0],
            [100.0, 100.2, 98.5, 98.0, 99.0],
            [100.0, 98.0, 97.0, 96.5, 97.5],
        ),
    ],
)
def test_partial_exit_occurs_next_bar_and_remainder_is_unchanged(
    side: float,
    opens: list[float],
    highs: list[float],
    lows: list[float],
) -> None:
    arrays = (
        np.asarray([opens], dtype=np.float32),
        np.asarray([highs], dtype=np.float32),
        np.asarray([lows], dtype=np.float32),
        np.asarray([opens], dtype=np.float32),
    )
    baseline = simulate_and_score(
        _row(side), *arrays, cost_pct=0.0, **_common()
    )
    p50 = simulate_and_score(
        _row(side),
        *arrays,
        cost_pct=0.0,
        partial_exit_on_first_trailing_activation_fraction=0.5,
        **_common(),
    )
    assert p50["partial_exit_mask"].tolist() == [True]
    # Bar 1 creates the favorable excursion; activation and the causal
    # executable partial occur at bar 2's open.
    assert p50["partial_exit_bars"].tolist() == [2]
    assert baseline["exit_reason"] == p50["exit_reason"]
    np.testing.assert_array_equal(baseline["exit_bars"], p50["exit_bars"])
    np.testing.assert_array_equal(
        baseline["gross_returns"], p50["gross_returns"]
    )
    expected = 0.5 * float(p50["partial_exit_returns"][0]) + 0.5 * float(
        baseline["gross_returns"][0]
    )
    assert float(p50["p50_gross_returns"][0]) == pytest.approx(expected)


def test_partial_exit_fee_formula_and_no_arm_timeout_parity() -> None:
    armed_paths = (
        np.array([[100.0, 100.0, 102.0, 102.0]], dtype=np.float32),
        np.array([[100.0, 102.0, 103.0, 102.5]], dtype=np.float32),
        np.array([[100.0, 99.8, 101.5, 101.0]], dtype=np.float32),
        np.array([[100.0, 101.8, 102.5, 101.5]], dtype=np.float32),
    )
    cost = 0.001
    p50 = simulate_and_score(
        _row(1.0, spread_bps=20.0),
        *armed_paths,
        cost_pct=cost,
        partial_exit_on_first_trailing_activation_fraction=0.5,
        **_common(),
    )
    partial_return = float(p50["partial_exit_returns"][0])
    remainder_return = float(p50["gross_returns"][0])
    expected_fee = (
        cost
        + 0.5 * (1.0 + partial_return) * cost
        + 0.5 * (1.0 + remainder_return) * cost
    )
    assert float(p50["p50_fee_returns"][0]) == pytest.approx(expected_fee)
    assert float(p50["p50_net_returns"][0]) == pytest.approx(
        float(p50["p50_gross_returns"][0]) - expected_fee
    )

    no_arm_paths = (
        np.array([[100.0, 100.0, 100.0]], dtype=np.float32),
        np.array([[100.0, 100.1, 100.1]], dtype=np.float32),
        np.array([[100.0, 99.9, 99.9]], dtype=np.float32),
        np.array([[100.0, 100.0, 100.0]], dtype=np.float32),
    )
    baseline = simulate_and_score(
        _row(1.0), *no_arm_paths, cost_pct=cost, **_common()
    )
    no_arm = simulate_and_score(
        _row(1.0),
        *no_arm_paths,
        cost_pct=cost,
        partial_exit_on_first_trailing_activation_fraction=0.5,
        **_common(),
    )
    assert not no_arm["partial_exit_mask"].any()
    np.testing.assert_array_equal(
        no_arm["p50_net_returns"], baseline["net_returns"]
    )


def test_partial_executes_at_open_before_same_bar_full_stop() -> None:
    # Bar 1 earns activation.  Bar 2 opens normally and then crosses the full
    # stop intrabar.  The partial must execute at bar-2 open first, while the
    # unchanged remainder still exits via the full stop.
    paths = (
        np.array([[100.0, 100.0, 100.0]], dtype=np.float32),
        np.array([[100.0, 102.0, 100.2]], dtype=np.float32),
        np.array([[100.0, 99.8, 90.0]], dtype=np.float32),
        np.array([[100.0, 101.5, 91.0]], dtype=np.float32),
    )
    common = _common()
    common["sl_mult"] = 2.0
    p50 = simulate_and_score(
        _row(1.0),
        *paths,
        cost_pct=0.0,
        partial_exit_on_first_trailing_activation_fraction=0.5,
        **common,
    )
    assert p50["partial_exit_mask"].tolist() == [True]
    assert p50["partial_exit_bars"].tolist() == [2]
    assert p50["exit_reason"] == ["full_sl"]
    assert p50["exit_bars"].tolist() == [2]
    assert float(p50["partial_exit_prices"][0]) > 0.0


def test_partial_fraction_validation() -> None:
    arrays = tuple(
        np.array([[100.0, 100.0]], dtype=np.float32) for _ in range(4)
    )
    with pytest.raises(ValueError, match="must be in"):
        simulate_and_score(
            _row(1.0),
            *arrays,
            partial_exit_on_first_trailing_activation_fraction=1.1,
        )
