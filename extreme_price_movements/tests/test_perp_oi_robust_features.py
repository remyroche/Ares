import numpy as np
import pandas as pd

from extreme_price_movements.config import CFG
from extreme_price_movements.features import (
    add_regime_gates,
    compute_features_hourly,
    compute_market_features,
)
from extreme_price_movements.features_oi import compute_oi_features
from extreme_price_movements.perp_features import compute_features


def test_robust_oi_change_features_do_not_require_volume():
    idx = pd.date_range("2026-01-01", periods=24 * 35, freq="h", tz="UTC")
    trend = np.linspace(0.0, 0.25, len(idx))
    seasonal = np.sin(np.linspace(0.0, 12.0, len(idx))) * 0.02
    open_interest = 1_000_000.0 * np.exp(trend + seasonal)
    close = 100.0 * np.exp(np.linspace(0.0, 0.05, len(idx)))

    df = pd.DataFrame(
        {
            "funding_rate": 0.00001,
            "open_interest": open_interest,
            "open_interest_quote": open_interest,
            "perp_price": close,
            "spot_price": close * 0.999,
            "mark_price": close,
            "close": close,
            "volume": 0.0,
            "quote_volume": 0.0,
        },
        index=idx,
    )

    out = compute_features(df)

    robust_cols = [
        "oi_value_log_1d_robust_z",
        "oi_value_log_7d_robust_z",
        "oi_chg_2h_robust_z",
        "oi_chg_4h_robust_z",
        "oi_chg_8h_robust_z",
    ]
    assert set(robust_cols).issubset(out.columns)
    assert out[robust_cols].iloc[24 * 8 :].notna().any().all()
    assert out[["oi_rel_vol_2h", "oi_rel_vol_4h", "oi_rel_vol_8h"]].isna().all().all()


def test_market_liquidation_lifecycle_oi_and_funding_features_are_emitted():
    idx = pd.date_range("2026-01-01", periods=24 * 45, freq="h", tz="UTC")
    cols = ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD", "XRP/USD:USD"]
    t = np.arange(len(idx), dtype=np.float32)
    crash_start = 24 * 20
    rebound_start = crash_start + 24
    rebound_end = rebound_start + 48

    price_log = np.log(100.0) + 0.00004 * t
    price_log[crash_start:rebound_start] += np.linspace(
        0.0, -0.16, rebound_start - crash_start, dtype=np.float32
    )
    price_log[rebound_start:rebound_end] += np.linspace(
        -0.16, -0.04, rebound_end - rebound_start, dtype=np.float32
    )
    price_log[rebound_end:] += -0.04

    oi_log = np.log(1_000_000.0) + 0.00003 * t
    oi_log[crash_start:rebound_start] += np.linspace(
        0.0, -0.18, rebound_start - crash_start, dtype=np.float32
    )
    oi_log[rebound_start:rebound_end] += np.linspace(
        -0.18, -0.36, rebound_end - rebound_start, dtype=np.float32
    )
    oi_log[rebound_end:] += -0.36

    price = pd.DataFrame(
        {
            col: np.exp(price_log + i * 0.01).astype(np.float32)
            for i, col in enumerate(cols)
        },
        index=idx,
    )
    open_interest = pd.DataFrame(
        {
            col: np.exp(oi_log + i * 0.02).astype(np.float32)
            for i, col in enumerate(cols)
        },
        index=idx,
    )
    quote_volume = pd.DataFrame(
        {
            col: (250_000.0 + 2_000.0 * np.sin(t / 12.0 + i)).astype(np.float32)
            for i, col in enumerate(cols)
        },
        index=idx,
    )
    funding = np.full(len(idx), 0.00018, dtype=np.float32)
    funding[rebound_start:] = np.linspace(
        0.00018, -0.00012, len(idx) - rebound_start, dtype=np.float32
    )
    funding_rate = pd.DataFrame(
        {col: (funding + i * 0.00001).astype(np.float32) for i, col in enumerate(cols)},
        index=idx,
    )
    # Missing funding for one asset must not suppress otherwise observable
    # OI/OHLCV liquidation-state composites for that asset.
    funding_rate["BTC/USD:USD"] = np.nan
    open_interest["BTC/USD:USD"] = np.nan

    out = compute_oi_features(
        oi_native=open_interest,
        price=price,
        quote_volume=quote_volume,
        funding_rate=funding_rate,
        bars_per_day=24,
    )

    required = [
        "mkt_oi_chg_1h",
        "pct_assets_oi_down_1h",
        "mkt_price_down_oi_down_1h",
        "mkt_price_up_oi_down_1h",
        "mkt_funding_mean",
        "negative_funding_x_price_up",
        "bars_since_mkt_oi_trough",
        "oi_drawdown_from_peak_24h",
        "oi_recovery_fraction_24h",
        "price_up_oi_down_1h_rz",
        "price_recovery_oi_still_falling_1h",
        "funding_crowding_release_4h",
        "mkt_pct_price_up_oi_down_1h",
        "mkt_oi_flush_breadth_recovery_4h",
        "asset_liquidation_phase_score",
        "asset_short_covering_score",
        "asset_mkt_liquidation_phase_divergence",
        "mkt_flush_exhaustion_score",
    ]
    assert set(required).issubset(out)
    for key in required:
        assert out[key].shape == price.shape
        assert out[key].iloc[24 * 8 :].notna().any().any()
    assert (
        out["mkt_price_down_oi_down_1h"].iloc[crash_start:rebound_start].max().max()
        > 0.0
    )
    assert (
        out["mkt_price_up_oi_down_1h"].iloc[rebound_start:rebound_end].max().max() > 0.0
    )
    assert (
        out["price_up_oi_down_1h_rz"].iloc[rebound_start:rebound_end].max().max() > 0.0
    )
    assert (
        out["mkt_oi_flush_breadth_recovery_4h"]
        .iloc[rebound_start:rebound_end]
        .max()
        .max()
        >= 0.0
    )
    assert (
        out["asset_liquidation_phase_score"]["BTC/USD:USD"].iloc[24 * 8 :].notna().any()
    )
    assert out["asset_short_covering_score"]["BTC/USD:USD"].iloc[24 * 8 :].notna().any()


def test_ohlcv_liquidation_lifecycle_features_are_emitted():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2026-01-01", periods=24 * 50, freq="h", tz="UTC")
    cols = ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD", "XRP/USD:USD"]
    base = np.cumsum(rng.normal(0.0, 0.01, size=(len(idx), len(cols))), axis=0)
    close = pd.DataFrame(100.0 * np.exp(base), index=idx, columns=cols)
    open_ = close.shift(1).fillna(close.iloc[0]) * (
        1.0 + rng.normal(0.0, 0.001, close.shape)
    )
    high = pd.DataFrame(
        np.maximum(open_.to_numpy(), close.to_numpy())
        * (1.0 + rng.uniform(0.0005, 0.01, close.shape)),
        index=idx,
        columns=cols,
    )
    low = pd.DataFrame(
        np.minimum(open_.to_numpy(), close.to_numpy())
        * (1.0 - rng.uniform(0.0005, 0.01, close.shape)),
        index=idx,
        columns=cols,
    )
    volume = pd.DataFrame(rng.lognormal(9.0, 0.5, close.shape), index=idx, columns=cols)
    panel = {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    requested = [
        "downside_deceleration_4h_rz",
        "price_recovery_from_low_24h_atr",
        "bars_since_price_low_24h_norm",
        "volume_climax_decay_4h",
        "range_climax_decay_4h",
        "wick_recovery_intensity",
        "market_pc1_variance_share_24h",
        "market_downside_pairwise_corr_24h",
        "market_breadth_recovery_from_24h_min",
        "market_pct_recovering_from_24h_low",
    ]
    cfg = dict(CFG)
    cfg.update(
        {
            "feature_portability_mode": "legacy",
            "feature_transform_cache_enabled": False,
            "feature_causal_transform_state_enabled": False,
            "live_causal_transform_state_enabled": False,
            "live_lgbm_mask_feature_fast_path_enabled": False,
        }
    )
    gates = add_regime_gates(
        compute_market_features(panel, cols, trend_sma_hours=24),
        cfg["gate_vol_lookback_hours"],
        cfg["gate_trend_thr"],
    )

    out, _, _ = compute_features_hourly(
        panel,
        gates,
        cfg,
        requested_feature_keys=requested,
    )

    assert set(requested).issubset(out)
    for key in requested:
        frame = out[key]
        assert frame.shape == close.shape
        assert np.isfinite(frame.iloc[24 * 3 :].to_numpy(dtype=np.float32)).any(), key
