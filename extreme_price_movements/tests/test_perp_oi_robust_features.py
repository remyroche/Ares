import numpy as np
import pandas as pd

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
