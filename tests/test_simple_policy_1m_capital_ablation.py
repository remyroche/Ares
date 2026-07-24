import numpy as np
import pandas as pd

from scripts.download_policy_execution_1m import _missing_buckets

from extreme_price_movements.simple_policy_1m_ablation import (
    FAMILY_A,
    FAMILY_B,
    FAMILY_CURRENT,
    MOD_MIN_CURRENT_GAP,
    REASON_FULL_SL,
    capacity_select,
    params_to_vector,
    simulate_1m_paths,
)


def _paths(n=1, bars=20):
    open0 = np.full(n, 100.0, dtype=np.float32)
    high = np.full((n, bars), 100.2, dtype=np.float32)
    low = np.full((n, bars), 99.8, dtype=np.float32)
    close = np.full((n, bars), 100.0, dtype=np.float32)
    return open0, high, low, close


def _run(family, params, modifiers=0, paths=None):
    open0, high, low, close = paths or _paths()
    n = len(open0)
    return simulate_1m_paths(
        np.arange(n, dtype=np.int64),
        open0,
        high,
        low,
        close,
        np.ones(n),
        np.full(n, 0.01),
        np.zeros(n),
        np.zeros(n),
        params_to_vector(params),
        family,
        modifiers,
        0.005,
        15.0,
        0.05,
        75.0,
    )


def test_current_zero_activation_disables_legacy_capital_protection():
    open0, high, low, close = _paths(bars=3)
    low[:] = 98.5
    result = _run(
        FAMILY_CURRENT,
        {
            "sl_mult": 1.0,
            "p1": 0.0,
            "trailing_activation_mult": 10.0,
            "trailing_power": 1.5,
            "trailing_squash_divisor": 2.0,
            "giveback_beta": 0.5,
        },
        paths=(open0, high, low, close),
    )
    assert result[4][0] == REASON_FULL_SL


def test_multilayer_envelope_is_not_zero_distance_at_entry():
    result = _run(
        FAMILY_B,
        {
            "sl_mult": 3.0,
            "trailing_activation_mult": 10.0,
            "trailing_power": 1.5,
            "trailing_squash_divisor": 2.0,
            "giveback_beta": 0.5,
            "p1": 2.0,
            "p2": 1.0,
            "p3": 1.0,
        },
    )
    assert result[0][0] == 19


def test_minimum_current_gap_uses_prior_close_and_does_not_look_at_current_close():
    open0, high, low, close = _paths(bars=4)
    high[0, 0] = 103.0
    close[0, 0] = 102.0
    low[0, 1] = 101.1
    close[0, 1] = 99.0
    result = _run(
        FAMILY_A,
        {
            "sl_mult": 3.0,
            "trailing_activation_mult": 10.0,
            "trailing_power": 1.5,
            "trailing_squash_divisor": 2.0,
            "giveback_beta": 0.5,
            "p1": 0.5,
            "min_current_gap_atr": 0.5,
        },
        modifiers=MOD_MIN_CURRENT_GAP,
        paths=(open0, high, low, close),
    )
    assert result[0][0] == 1


def test_capacity_enforces_two_new_and_eight_open_with_symbol_uniqueness():
    timestamps = np.array([0] * 12, dtype=np.int64)
    symbols = np.arange(12, dtype=np.int32)
    exits = np.full(12, 60, dtype=np.int32)
    selected = capacity_select(timestamps, symbols, exits, 1, 8, 2)
    assert selected.sum() == 2

    timestamps = np.arange(10, dtype=np.int64) * 60 * 1_000_000_000
    symbols = np.zeros(10, dtype=np.int32)
    selected = capacity_select(timestamps, symbols, exits[:10], 1, 8, 2)
    assert selected.sum() == 1


def test_missing_bucket_merges_disjoint_candidate_windows_in_same_day():
    windows = [
        (pd.Timestamp("2026-05-01 01:00", tz="UTC"), pd.Timestamp("2026-05-01 03:00", tz="UTC")),
        (pd.Timestamp("2026-05-01 10:00", tz="UTC"), pd.Timestamp("2026-05-01 12:00", tz="UTC")),
    ]
    buckets = _missing_buckets(windows, pd.DatetimeIndex([], tz="UTC"), 1440)
    assert buckets == [(windows[0][0], windows[1][1])]
