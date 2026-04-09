import numpy as np
import pandas as pd

from extreme_price_movements.triad_targets import (

    apply_surprisal_to_targets,
    compute_rolling_surprisal,
    get_bounded_triad,
)


def test_compute_rolling_surprisal_emits_tail_scores() -> None:
    x = np.array([0.5, 0.5, 0.5, 0.5, 0.95, 0.5, 0.05], dtype=np.float64)
    surprisal = compute_rolling_surprisal(
        x, lookback=4, min_samples=3, smooth_window=None
    )

    assert np.isnan(surprisal[0])
    assert np.isfinite(surprisal[4])
    assert np.isfinite(surprisal[6])
    assert surprisal[4] > 0.0
    assert surprisal[6] > 0.0


def test_apply_surprisal_to_targets_is_symbol_local() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["A"] * 6 + ["B"] * 6,
            "target_eff": [0.5, 0.5, 0.5, 0.95, 0.5, 0.5, 0.5, 0.5, 0.5, 0.05, 0.5, 0.5],
        }
    )

    out = apply_surprisal_to_targets(
        df,
        target_cols=["target_eff"],
        lookback=4,
        min_samples=3,
        smooth_window=None,
        blend_weight=0.2,
        reference_bits=3.0,
    )

    assert "target_eff_surprisal" in out.columns
    # The rare event in each symbol should be scored independently.
    assert out.loc[3, "target_eff_surprisal"] > 0.0
    assert out.loc[9, "target_eff_surprisal"] > 0.0
    # Early rows without enough history should remain NaN in the companion feature.
    assert np.isnan(out.loc[0, "target_eff_surprisal"])
    # Blending is weak and stays bounded.
    assert float(out["target_eff"].min()) >= 0.0
    assert float(out["target_eff"].max()) <= 1.0


def test_get_bounded_triad_adds_surprisal_features_and_blends_targets() -> None:
    n = 240
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    close = np.linspace(100.0, 130.0, n) + 0.5 * np.sin(np.arange(n) / 9.0)
    high = close + 1.0
    low = close - 1.0
    volume = 1000.0 + 20.0 * np.cos(np.arange(n) / 7.0)
    atr = np.full(n, 1.5, dtype=np.float64)

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": ["AAA"] * n,
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "atr": atr,
        }
    )

    base = get_bounded_triad(
        df.copy(),
        n=12,
        use_surprisal_selectivity=False,
    )
    out = get_bounded_triad(
        df.copy(),
        n=12,
        use_surprisal_selectivity=True,
        surprisal_lookback=40,
        surprisal_min_samples=10,
        surprisal_smooth_window=None,
        surprisal_blend_weight=0.2,
    )

    for col in ["target_eff_surprisal", "target_vame_surprisal"]:
        assert col in out.columns
        assert np.isfinite(out[col].to_numpy(dtype=np.float64)[20:]).any()

    for col in ["target_eff", "target_vame"]:
        out_arr = out[col].to_numpy(dtype=np.float64)
        base_arr = base[col].to_numpy(dtype=np.float64)
        assert np.nanmin(out_arr) >= 0.0
        assert np.nanmax(out_arr) <= 1.0
        assert np.any(np.abs(out_arr - base_arr) > 1e-8)


def test_get_bounded_triad_respects_symbol_boundaries() -> None:
    n_per_symbol = 32
    horizon = 4
    idx = pd.date_range("2025-01-01", periods=n_per_symbol * 2, freq="h", tz="UTC")
    close_a = np.linspace(100.0, 120.0, n_per_symbol)
    close_b = np.linspace(400.0, 430.0, n_per_symbol)
    close = np.r_[close_a, close_b]

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": ["AAA"] * n_per_symbol + ["BBB"] * n_per_symbol,
            "close": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": np.full(n_per_symbol * 2, 1000.0, dtype=np.float64),
            "atr": np.full(n_per_symbol * 2, 1.5, dtype=np.float64),
        }
    )

    out = get_bounded_triad(
        df,
        n=horizon,
        min_history_percentile=1,
        use_surprisal_selectivity=False,
    )

    for symbol in ("AAA", "BBB"):
        sym = out.loc[out["symbol"] == symbol]
        assert sym["target_eff"].tail(horizon).isna().all()


def test_get_bounded_triad_has_high_valid_rate_on_long_multi_symbol_panel() -> None:
    n_per_symbol = 5000
    horizon = 12
    idx = pd.date_range("2025-01-01", periods=n_per_symbol * 2, freq="h", tz="UTC")
    t = np.arange(n_per_symbol, dtype=np.float64)
    close_a = 100.0 + 0.02 * t + 0.8 * np.sin(t / 15.0)
    close_b = 300.0 + 0.03 * t + 1.1 * np.cos(t / 17.0)
    close = np.r_[close_a, close_b]

    df = pd.DataFrame(
        {
            "timestamp": idx,
            "symbol": ["AAA"] * n_per_symbol + ["BBB"] * n_per_symbol,
            "close": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": 1000.0 + 50.0 * np.sin(np.arange(n_per_symbol * 2) / 11.0),
            "atr": np.full(n_per_symbol * 2, 1.5, dtype=np.float64),
        }
    )

    out = get_bounded_triad(
        df,
        n=horizon,
        min_history_percentile=1,
        use_surprisal_selectivity=False,
    )

    eff_valid_rate = float(np.isfinite(out["target_eff"]).mean())
    vame_valid_rate = float(np.isfinite(out["target_vame"]).mean())

    assert eff_valid_rate > 0.99
    assert vame_valid_rate > 0.99


