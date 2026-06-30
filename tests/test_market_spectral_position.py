import numpy as np
import pandas as pd

from extreme_price_movements.performance_regimes.spectral_position import (
    MarketSpectralPositionConfig,
    fit_market_spectral_position_encoder,
    transform_market_spectral_position,
)


def _train_frame(rows: int = 40) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    x = np.linspace(-1.0, 1.0, rows)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "fs__mkt_ret_eq_1h__mean": x,
            "fs__market_breadth_1h__median": 0.7 * x + 0.1 * np.sin(np.arange(rows)),
            "fs__rv_24h__median": np.cos(np.arange(rows) / 5.0),
            "fs__mkt_oi_chg_z_24h__median": np.sin(np.arange(rows) / 4.0),
        }
    )


def test_spectral_position_uses_prior_matrix_but_current_projection():
    train = _train_frame()
    cfg = MarketSpectralPositionConfig(lookback=24, min_periods=12, top_k=3, max_features=4)
    encoder = fit_market_spectral_position_encoder(train, config=cfg)
    ts = pd.Timestamp("2026-01-03 00:00", tz="UTC")
    normal = pd.DataFrame(
        {
            "timestamp": [ts],
            "fs__mkt_ret_eq_1h__mean": [0.1],
            "fs__market_breadth_1h__median": [0.05],
            "fs__rv_24h__median": [0.2],
            "fs__mkt_oi_chg_z_24h__median": [0.1],
        }
    )
    shocked = normal.copy()
    shocked["fs__mkt_ret_eq_1h__mean"] = 20.0

    normal_out = transform_market_spectral_position(normal, encoder)
    shocked_out = transform_market_spectral_position(shocked, encoder)

    for col in [
        "state_spectral_eig_lambda1_share",
        "state_spectral_eig_top3_share",
        "state_spectral_eig_effective_rank",
        "state_spectral_eig_entropy",
        "state_spectral_eig_condition",
    ]:
        assert float(normal_out[col].iloc[0]) == float(shocked_out[col].iloc[0])
    assert float(normal_out["state_spectral_pc1_score"].iloc[0]) != float(
        shocked_out["state_spectral_pc1_score"].iloc[0]
    )
    assert float(shocked_out["state_spectral_top3_mahalanobis"].iloc[0]) > float(
        normal_out["state_spectral_top3_mahalanobis"].iloc[0]
    )


def test_spectral_position_features_are_finite_float32():
    train = _train_frame()
    encoder = fit_market_spectral_position_encoder(
        train,
        config=MarketSpectralPositionConfig(lookback=12, min_periods=6, top_k=3),
    )
    out = transform_market_spectral_position(train.iloc[:8].copy(), encoder)

    spectral_cols = [c for c in out.columns if c.startswith("state_spectral_")]
    assert spectral_cols
    assert all(str(out[c].dtype) == "float32" for c in spectral_cols)
    assert np.isfinite(out[spectral_cols].to_numpy()).all()
