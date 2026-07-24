import pytest
import numpy as np
import pandas as pd
from extreme_price_movements.feature_transforms import CausalFeatureTransformer

def reference_risk_normalized_transform(df, window, sigma_k):
    """Independent pandas reference for the risk-normalized family."""
    input_bad = ~np.isfinite(df.to_numpy(dtype=np.float64, copy=False))
    out = np.arcsinh(df)
    mu = out.rolling(window=window, min_periods=1).mean()
    sigma = out.rolling(window=window, min_periods=1).std(ddof=1)
    z = (out - mu) / (sigma + 1e-12)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    z = z.clip(lower=-sigma_k, upper=sigma_k)
    values = z.to_numpy(dtype=np.float64, copy=False)
    values[input_bad] = np.nan
    z.iloc[:, :] = values
    return z

def test_causal_feature_transformer_equivalence():
    np.random.seed(42)
    rows = 1000
    cols = 5
    data = np.random.randn(rows, cols)
    # Add some NaNs
    data[10:20, 0] = np.nan
    data[50:60, 2] = np.nan

    df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(cols)])

    window = 50
    winsor_qt = 0.02

    transformer = CausalFeatureTransformer(winsor_qt=winsor_qt, roll_window=window)

    # ``ret4h`` is registered as a risk-normalized continuous feature. Unknown
    # names are deliberately treated as already standardized and must not be
    # compared to this transform path.
    res_opt = transformer.transform(df.copy(), name="ret4h")

    res_ref = reference_risk_normalized_transform(
        df.copy(), window, transformer.sigma_k
    )

    pd.testing.assert_frame_equal(
        res_opt.astype(np.float64), res_ref, atol=1e-4, check_dtype=False
    )


def test_transform_batch_preserves_mixed_2d_widths():
    idx = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    wide = pd.DataFrame(
        np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(8, 3),
        index=idx,
        columns=["A", "B", "C"],
    )
    narrow = pd.DataFrame(
        np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(8, 1),
        index=idx,
        columns=["MKT"],
    )
    feats = {
        "loc_range_pos_24": wide.copy(),
        "prior_volatility": narrow.copy(),
    }
    transformer = CausalFeatureTransformer(enable_cache=False, roll_window=4)

    out = transformer.transform_batch(feats, skip_keys=set(), chunk_size=10)

    assert out["loc_range_pos_24"].shape == (8, 3)
    assert out["prior_volatility"].shape == (8, 1)
    assert np.nanstd(out["loc_range_pos_24"]) > 0.0
    assert np.nanstd(out["prior_volatility"]) > 0.0


if __name__ == "__main__":
    test_causal_feature_transformer_equivalence()
